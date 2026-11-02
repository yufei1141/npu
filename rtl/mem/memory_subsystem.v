//=============================================================================
// memory_subsystem : scratchpad banks, per-client routing and banking
//
//   Follows docs/npu_integration/05_memory_subsystem.md.
//
//   Five fixed clients: 0 MPU_A, 1 VPU, 2 SFU, 3 RESULT_WB, 4 DMA_CORE.
//   Each client has one request and one response channel and at most one
//   outstanding request.  A request is routed to the bank of its logical
//   buffer.  Each physical bank has its own round-robin arbiter, one RAM
//   access in flight and one response holding register.
//
//   Requests are checked here as a second line of defence (the scheduler also
//   validates capacity): unknown/disabled buffer, unaligned byte offset and
//   an access that crosses the buffer end all produce an error response and
//   never reach the RAM.  Every accepted request gets exactly one response,
//   including error responses; the response is held until the client takes
//   it, and the client stays busy until then.
//
//   A routing error (buffer or bank id outside the implemented set) has no
//   bank to wait on, so it uses a per-client error holding register.
//=============================================================================
//=============================================================================
// 中文说明
//
//   片内 scratchpad：8 个物理 bank、按逻辑 buffer 路由、每个 bank 独立轮询
//   仲裁。实现依据 docs/npu_integration/05_memory_subsystem.md。
//
//   客户端固定五个：0 MPU_A、1 VPU、2 SFU、3 RESULT_WB、4 DMA_CORE。每个
//   客户端只有一组请求通道和一组响应通道，**最多一个请求在途**——这是本
//   设计的简化点：它保证响应永远不会"张冠李戴"，代价是每个客户端无法流水
//   地连发多个访问。
//
//   路由与仲裁：
//     - 请求按逻辑 buffer 号路由到对应物理 bank；
//     - 每个 bank 有自己的轮询（round-robin）仲裁器、一个在途 RAM 访问和
//       一个响应保持寄存器，避免某个客户端长期霸占某个 bank。
//
//   第二道校验（第一道在调度器）：
//     - 未知/未启用的 buffer、非对齐的字节偏移、跨越 buffer 尾部的访问，
//       都在这里直接产生错误响应，**不会到达 RAM**；
//     - 每个被接受的请求都恰好得到一个响应（包括错误响应），响应一直保持
//       到客户端取走为止，期间客户端保持 busy。
//
//   边界情形：
//     - 路由类错误（buffer 号或 bank 号超出实现范围）根本没有 bank 可以等，
//       所以走每个客户端独立的错误保持寄存器返回。
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

`include "npu_defs.vh"

module memory_subsystem #(
    parameter MEM_DATA_W = 256,
    parameter BANKS      = 8,
    parameter BANK_DEPTH = 1024,
    parameter L_RAM      = 1
)(
    input  wire                    clk_core,
    input  wire                    rst_core_n,

    input  wire [7:0]              buf_enable,
    input  wire [63:0]             buf_bank_flat,
    input  wire [255:0]            buf_base_flat,
    input  wire [255:0]            buf_size_flat,

    input  wire [4:0]              c_req_valid,
    output wire [4:0]              c_req_ready,
    input  wire [5*`NPU_REQ_W(MEM_DATA_W, MEM_DATA_W/8)-1:0] c_req,

    output wire [4:0]              c_rsp_valid,
    input  wire [4:0]              c_rsp_ready,
    output wire [5*`NPU_RSP_W(MEM_DATA_W)-1:0] c_rsp,

    output wire [15:0]             outstanding_count,
    output reg                     access_fault_valid,
    output reg  [63:0]             access_fault_info,

    output wire [BANKS-1:0]        bank_busy
);

    localparam DW    = MEM_DATA_W;
    localparam B     = MEM_DATA_W / 8;
    localparam [32:0] BU = MEM_DATA_W / 8;   // byte size, sized for end = off + BU
    localparam LOG2B = $clog2(B);
    localparam REQ_W = `NPU_REQ_W(MEM_DATA_W, MEM_DATA_W/8);
    localparam RSP_W = `NPU_RSP_W(MEM_DATA_W);
    localparam AW    = $clog2(BANK_DEPTH);

    localparam B_EMPTY = 2'd0;
    localparam B_RAM   = 2'd1;
    localparam B_HOLD  = 2'd2;

    integer c, b, k, idx;

    // ------------------------------------------------------------------
    // per-client request decode
    // ------------------------------------------------------------------
    reg          rq_write [0:4];
    reg  [7:0]   rq_buf   [0:4];
    reg  [31:0]  rq_off   [0:4];
    reg  [DW-1:0] rq_data [0:4];
    reg  [B-1:0] rq_strb  [0:4];
    reg  [7:0]   rq_tag   [0:4];

    reg          rq_route [0:4];   // buffer/bank usable
    reg          rq_range [0:4];
    reg          rq_align [0:4];
    reg  [7:0]   rq_reason[0:4];
    reg  [2:0]   rq_bank  [0:4];
    reg  [AW-1:0] rq_word [0:4];

    // per-bank combinbational request / grant
    reg  [4:0]   bank_req  [0:BANKS-1];
    reg          bank_gnt  [0:BANKS-1];
    reg  [2:0]   bank_sel  [0:BANKS-1];
    reg  [4:0]   client_ready;
    reg  [4:0]   rsp_valid;
    reg  [5*RSP_W-1:0] rsp_flat;

    // bank state
    reg  [1:0]   bstate   [0:BANKS-1];
    reg  [2:0]   bowner   [0:BANKS-1];
    reg  [2:0]   brr      [0:BANKS-1];
    reg  [7:0]   btag     [0:BANKS-1];
    reg  [DW-1:0] bdata   [0:BANKS-1];
    reg          berr     [0:BANKS-1];
    reg  [7:0]   breason  [0:BANKS-1];

    // client outstanding / bypass error response
    reg  [4:0]   client_busy;
    reg  [4:0]   err_hold_valid;
    reg  [5*RSP_W-1:0] err_hold;

    // RAM ports
    wire [BANKS-1:0]      ram_en;
    wire [BANKS-1:0]      ram_we;
    wire [BANKS*AW-1:0]   ram_addr;
    wire [BANKS*DW-1:0]   ram_wdata;
    wire [BANKS*B-1:0]    ram_wstrb;
    wire [BANKS-1:0]      ram_rvalid;
    wire [BANKS*DW-1:0]   ram_rdata;
    wire [BANKS-1:0]      ram_error;

    genvar gb;
    generate
        for (gb = 0; gb < BANKS; gb = gb + 1) begin : g_bank
            sram_wrapper #(
                .MEM_DATA_W(MEM_DATA_W), .DEPTH(BANK_DEPTH), .L_RAM(L_RAM)
            ) u_ram (
                .clk(clk_core), .rst_n(rst_core_n),
                .en(ram_en[gb]), .we(ram_we[gb]),
                .addr(ram_addr[gb*AW +: AW]),
                .wdata(ram_wdata[gb*DW +: DW]),
                .wstrb(ram_wstrb[gb*B +: B]),
                .rvalid(ram_rvalid[gb]),
                .rdata(ram_rdata[gb*DW +: DW]),
                .error(ram_error[gb]),
                .err_inj(1'b0)
            );
        end
    endgenerate

    // ------------------------------------------------------------------
    // routing / legality / arbitration (combinational)
    // ------------------------------------------------------------------
    reg [32:0] end33;

    always @* begin
        client_ready  = 5'd0;
        rsp_valid     = 5'd0;
        rsp_flat      = {(5*RSP_W){1'b0}};

        for (c = 0; c < 5; c = c + 1) begin
            rq_write[c] = c_req[c*REQ_W + 0];
            rq_buf[c]   = c_req[c*REQ_W + 1 +: 8];
            rq_off[c]   = c_req[c*REQ_W + 9 +: 32];
            rq_data[c]  = c_req[c*REQ_W + 41 +: DW];
            rq_strb[c]  = c_req[c*REQ_W + 41 + DW +: B];
            rq_tag[c]   = c_req[c*REQ_W + 41 + DW + B +: 8];

            rq_route[c]  = (rq_buf[c] < 8) && buf_enable[rq_buf[c]];
            rq_bank[c]   = (rq_buf[c] < 8) ?
                           buf_bank_flat[8*rq_buf[c] +: 8] : 3'd7;
            rq_route[c]  = rq_route[c] && (rq_bank[c] < BANKS);
            rq_align[c]  = (rq_off[c][LOG2B-1:0] == {LOG2B{1'b0}});
            end33        = {1'b0, rq_off[c]} + BU;
            rq_range[c]  = (rq_buf[c] < 8) ?
                           (end33 <= {1'b0, buf_size_flat[32*rq_buf[c] +: 32]})
                           : 1'b0;
            rq_word[c]   = (rq_buf[c] < 8) ?
                           (buf_base_flat[32*rq_buf[c] +: 32] + (rq_off[c] >> LOG2B))
                           : {AW{1'b0}};

            if (!rq_route[c])          rq_reason[c] = `NPU_MEMR_BUF;
            else if (!rq_align[c])     rq_reason[c] = `NPU_MEMR_ALIGN;
            else if (!rq_range[c])     rq_reason[c] = `NPU_MEMR_RANGE;
            else                       rq_reason[c] = `NPU_MEMR_OK;
        end

        // requests that can be routed to a bank
        for (b = 0; b < BANKS; b = b + 1) bank_req[b] = 5'd0;
        for (c = 0; c < 5; c = c + 1) begin
            if (c_req_valid[c] && !client_busy[c] && rq_route[c]) begin
                for (b = 0; b < BANKS; b = b + 1) begin
                    if (rq_bank[c] == b[2:0])
                        bank_req[b][c] = 1'b1;
                end
            end
        end

        // one round-robin grant per bank; a busy bank grants nothing, so its
        // clients never see a ready that cannot be honoured
        for (b = 0; b < BANKS; b = b + 1) begin
            bank_gnt[b] = 1'b0;
            bank_sel[b] = 3'd0;
            if (bstate[b] == B_EMPTY) begin
                for (k = 0; k < 5; k = k + 1) begin
                    idx = (brr[b] + k) % 5;
                    if (!bank_gnt[b] && bank_req[b][idx]) begin
                        bank_gnt[b] = 1'b1;
                        bank_sel[b] = idx[2:0];
                    end
                end
            end
        end

        // client ready: routed grant, or immediate error bypass
        for (c = 0; c < 5; c = c + 1) begin
            if (client_busy[c])
                client_ready[c] = 1'b0;
            else if (!rq_route[c])
                client_ready[c] = 1'b1;
            else
                client_ready[c] = bank_gnt[rq_bank[c]] &&
                                  (bank_sel[rq_bank[c]] == c[2:0]);
        end

        // response mux: bypass errors first, then the owning bank
        for (c = 0; c < 5; c = c + 1) begin
            if (err_hold_valid[c]) begin
                rsp_valid[c] = 1'b1;
                rsp_flat[c*RSP_W +: RSP_W] = err_hold[c*RSP_W +: RSP_W];
            end
            for (b = 0; b < BANKS; b = b + 1) begin
                if ((bstate[b] == B_HOLD) && (bowner[b] == c[2:0])) begin
                    rsp_valid[c] = 1'b1;
                    rsp_flat[c*RSP_W +: RSP_W] =
                        {breason[b], berr[b], btag[b], bdata[b]};
                end
            end
        end
    end

    assign c_req_ready = client_ready;
    assign c_rsp_valid = rsp_valid;
    assign c_rsp       = rsp_flat;

    // ------------------------------------------------------------------
    // RAM request drive: a legal grant issues immediately
    // ------------------------------------------------------------------
    wire [BANKS-1:0] bank_fire;
    wire [BANKS-1:0] bank_legal;

    generate
        for (gb = 0; gb < BANKS; gb = gb + 1) begin : g_ram_drive
            assign bank_fire[gb]  = (bstate[gb] == B_EMPTY) && bank_gnt[gb];
            assign bank_legal[gb] = rq_route[bank_sel[gb]] &&
                                    rq_align[bank_sel[gb]] &&
                                    rq_range[bank_sel[gb]];
            assign ram_en[gb]     = bank_fire[gb] && bank_legal[gb];
            assign ram_we[gb]     = rq_write[bank_sel[gb]];
            assign ram_addr[gb*AW +: AW] = rq_word[bank_sel[gb]][AW-1:0];
            assign ram_wdata[gb*DW +: DW] = rq_data[bank_sel[gb]];
            assign ram_wstrb[gb*B +: B]  = rq_strb[bank_sel[gb]];
        end
    endgenerate

    // ------------------------------------------------------------------
    // sequential state
    // ------------------------------------------------------------------
    integer b2, c2;

    always @(posedge clk_core or negedge rst_core_n) begin
        if (!rst_core_n) begin
            client_busy      <= 5'd0;
            err_hold_valid   <= 5'd0;
            err_hold         <= {(5*RSP_W){1'b0}};
            access_fault_valid <= 1'b0;
            access_fault_info  <= 64'd0;
            for (b2 = 0; b2 < BANKS; b2 = b2 + 1) begin
                bstate[b2]  <= B_EMPTY;
                bowner[b2]  <= 3'd0;
                brr[b2]     <= 3'd0;
                btag[b2]    <= 8'd0;
                bdata[b2]   <= {DW{1'b0}};
                berr[b2]    <= 1'b0;
                breason[b2] <= 8'd0;
            end
        end else begin
            access_fault_valid <= 1'b0;

            // routing-error bypass: no bank involved
            for (c2 = 0; c2 < 5; c2 = c2 + 1) begin
                if (c_req_valid[c2] && client_ready[c2] && !rq_route[c2]) begin
                    client_busy[c2]    <= 1'b1;
                    err_hold_valid[c2] <= 1'b1;
                    err_hold[c2*RSP_W +: RSP_W] <=
                        {rq_reason[c2], 1'b1, rq_tag[c2], {DW{1'b0}}};
                    access_fault_valid <= 1'b1;
                    access_fault_info  <= {13'd0, c2[2:0], rq_buf[c2],
                                           rq_off[c2], rq_reason[c2]};
                end
            end

            // client response accepted
            for (c2 = 0; c2 < 5; c2 = c2 + 1) begin
                if (rsp_valid[c2] && c_rsp_ready[c2]) begin
                    client_busy[c2]    <= 1'b0;
                    err_hold_valid[c2] <= 1'b0;
                end
            end

            // banks
            for (b2 = 0; b2 < BANKS; b2 = b2 + 1) begin
                case (bstate[b2])
                B_EMPTY: begin
                    if (bank_gnt[b2]) begin
                        bowner[b2] <= bank_sel[b2];
                        btag[b2]   <= rq_tag[bank_sel[b2]];
                        brr[b2]    <= (bank_sel[b2] + 3'd1) % 5;
                        client_busy[bank_sel[b2]] <= 1'b1;
                        if (bank_legal[b2]) begin
                            bstate[b2] <= B_RAM;
                        end else begin
                            bdata[b2]   <= {DW{1'b0}};
                            berr[b2]    <= 1'b1;
                            breason[b2] <= rq_reason[bank_sel[b2]];
                            access_fault_valid <= 1'b1;
                            access_fault_info  <= {13'd0, bank_sel[b2],
                                                   rq_buf[bank_sel[b2]],
                                                   rq_off[bank_sel[b2]],
                                                   rq_reason[bank_sel[b2]]};
                            bstate[b2] <= B_HOLD;
                        end
                    end
                end
                B_RAM: begin
                    if (ram_rvalid[b2]) begin
                        bdata[b2]   <= ram_rdata[b2*DW +: DW];
                        berr[b2]    <= ram_error[b2];
                        breason[b2] <= ram_error[b2] ? `NPU_MEMR_RANGE
                                                     : `NPU_MEMR_OK;
                        bstate[b2]  <= B_HOLD;
                    end
                end
                B_HOLD: begin
                    if (c_rsp_ready[bowner[b2]])
                        bstate[b2] <= B_EMPTY;
                end
                endcase
            end
        end
    end

    // ------------------------------------------------------------------
    // diagnostics
    // ------------------------------------------------------------------
    // number of clients with a request in flight (0..5)
    assign outstanding_count = {11'd0, client_busy[0]} +
                               {11'd0, client_busy[1]} +
                               {11'd0, client_busy[2]} +
                               {11'd0, client_busy[3]} +
                               {11'd0, client_busy[4]};

    genvar gb2;
    generate
        for (gb2 = 0; gb2 < BANKS; gb2 = gb2 + 1) begin : g_busy
            assign bank_busy[gb2] = (bstate[gb2] != B_EMPTY);
        end
    endgenerate

endmodule

`default_nettype wire
