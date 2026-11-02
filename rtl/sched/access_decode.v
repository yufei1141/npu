//=============================================================================
// access_decode : combinational decode of a latched command
//
//   Turns the 512-bit command into the scheduler's view of it:
//     - engine class (DMA / MPU / VPU / SFU / FENCE / RELEASE)
//     - scratchpad read and write sets, with the byte extent of every access
//     - header fields the scheduler needs (job, dependency, event slot, flags)
//     - buffer configuration checks: enabled, capacity and within-command
//       physical aliasing between a read range and a write range
//
//   This is the "resource" half of the scheduler's CHECK state.  Static
//   instruction checks (opcode/field/alignment) belong to the command frontend
//   and arrive as issue_error; this module only looks at buffer resources.
//
//   Buffer configuration is flattened because Verilog-2001 has no array
//   ports:   bank[b] = buf_bank_flat[8*b +: 8], and likewise for base/size.
//   Byte offset is relative to the buffer; the physical byte range is
//   base_word[b]*B + offset.
//
//   The byte extent of a command is the span from the first to the last byte
//   actually touched.  Broadcasting shrinks the src1 extent to 1 element
//   (scalar) or ceil(count/2) elements (pair).  INPLACE removes src0 from the
//   read set: the shared buffer is taken once as an exclusive write.
//=============================================================================
//=============================================================================
// 中文说明
//
//   对已经锁存的命令做纯组合译码，把 512 位命令翻译成调度器关心的视图：
//     - 引擎类别（DMA / MPU / VPU / SFU / FENCE / RELEASE）；
//     - scratchpad 的读集合与写集合，以及每段访问的**字节范围**；
//     - 调度器需要的头部字段（job、依赖、事件槽、flags）；
//     - buffer 配置检查：是否启用、容量是否够、以及同一条命令内部的读区间
//       与写区间是否在物理上重叠（aliasing）。
//
//   职责边界：这是调度器 CHECK 状态里"资源那一半"。指令本身的静态检查
//   （opcode、字段、对齐）属于 command_frontend，以 issue_error 的形式传进来；
//   本模块只看 buffer 资源，不重复做指令合法性判断。
//
//   为什么有 *_flat 端口：Verilog-2001 不支持数组端口，所以 buffer 配置被摊平
//   成向量——bank[b] = buf_bank_flat[8*b +: 8]，base/size 同理。这是语言限制，
//   不是设计意图。
//
//   地址换算：命令里的字节偏移是**相对 buffer 的**，物理字节范围是
//   base_word[b]*B + offset，B 是一个存储字的字节数。
//
//   字节范围（extent）的定义是"实际被触及的第一个字节到最后一个字节"：
//     - 广播（broadcast）时 src1 的范围缩小——scalar 只有 1 个元素，pair 是
//       ceil(count/2) 个元素；
//     - INPLACE 会把 src0 从读集合里去掉：共享的那个 buffer 只以一次独占写
//       的身份参与冲突检查，否则它自己就会和自己冲突。
//
//   所以这里的判据是"保守但正确"：范围取首尾跨度，中间的缝隙也算占用，宁可
//   少并发也不出错。
//=============================================================================
`timescale 1ns/1ps
`default_nettype none

`include "npu_defs.vh"

module access_decode #(
    parameter MEM_DATA_W  = 256,
    parameter EVENT_SLOTS = 16
)(
    input  wire [511:0] cmd,

    input  wire [7:0]   buf_enable,
    input  wire [63:0]  buf_bank_flat,
    input  wire [255:0] buf_base_flat,
    input  wire [255:0] buf_size_flat,

    output reg  [2:0]   kind,        // 0 illegal, 1 DMA, 2 MPU, 3 VPU, 4 SFU,
                                     // 5 FENCE, 6 RELEASE
    output reg  [7:0]   opcode,
    output reg  [7:0]   subop,
    output reg  [15:0]  job,
    output reg  [15:0]  dep_job,
    output reg  [7:0]   dep_slot,
    output reg  [7:0]   ev_slot,

    output reg          dep_en,
    output reg          inplace,
    output reg          irq,

    output reg  [7:0]   read_mask,
    output reg  [7:0]   write_mask,

    output reg  [7:0]   rel_target_slot,
    output reg  [15:0]  rel_target_job,

    output reg  [15:0]  chk_status
);

    localparam K_NONE    = 3'd0;
    localparam K_DMA     = 3'd1;
    localparam K_MPU     = 3'd2;
    localparam K_VPU     = 3'd3;
    localparam K_SFU     = 3'd4;
    localparam K_FENCE   = 3'd5;
    localparam K_RELEASE = 3'd6;

    localparam B = MEM_DATA_W / 8;

    // ------------------------------------------------------------------
    // command fields
    // ------------------------------------------------------------------
    wire [7:0]  f_op    = cmd[15:8];
    wire [7:0]  f_sub   = cmd[23:16];
    wire [7:0]  f_flags = cmd[31:24];
    wire [7:0]  f_src0b = cmd[71:64];
    wire [7:0]  f_src1b = cmd[79:72];
    wire [7:0]  f_dstb  = cmd[87:80];
    wire [7:0]  f_fmt   = cmd[95:88];

    wire [7:0]  f_fs0 = {6'd0, f_fmt[1:0]};
    wire [7:0]  f_fs1 = {6'd0, f_fmt[3:2]};
    wire [7:0]  f_fd  = {6'd0, f_fmt[5:4]};

    wire [31:0] p0  = cmd[159:128];
    wire [31:0] p1  = cmd[191:160];
    wire [31:0] p2  = cmd[223:192];
    wire [31:0] p3  = cmd[255:224];
    wire [31:0] p4  = cmd[287:256];
    wire [31:0] p5  = cmd[319:288];
    wire [31:0] p6  = cmd[351:320];
    wire [31:0] p7  = cmd[383:352];
    wire [31:0] p8  = cmd[415:384];
    wire [31:0] p9  = cmd[447:416];

    wire [31:0] es0 = `NPU_ELEM_BYTES(f_fs0);
    wire [31:0] es1 = `NPU_ELEM_BYTES(f_fs1);
    wire [31:0] esd = `NPU_ELEM_BYTES(f_fd);

    integer  b, r, w;
    reg [7:0]  use_r, use_w;
    reg [31:0] off   [0:7];
    reg [32:0] len   [0:7];     // 33 bit so a 32-bit overflow is visible

    reg [63:0] t64, t64b, t64c;
    reg [39:0] endb;
    reg        ovf;

    reg [31:0] count, m_len, kvalid, nvalid, rows, cols, g8, s4;
    reg [31:0] dlo, dhi;
    reg [32:0] ecnt;

    // Extend the byte span of one buffer access.  A command may touch the
    // same logical buffer more than once (shared read of src0/src1), so the
    // stored range is the union of all accesses: [off, off+len).
    reg [39:0] alo, ahi, nhi;

    task span;
        input [7:0]  id;
        input [31:0] o;
        input [32:0] l;
        begin
            if (id >= 8) begin
                chk_status = `NPU_ST_BAD_FIELD;
            end else if (!(use_r[id] || use_w[id])) begin
                off[id] = o;
                len[id] = l;
            end else begin
                alo = {8'd0, off[id]};
                ahi = alo + {7'd0, len[id]};
                nhi = {8'd0, o} + {7'd0, l};
                if ({8'd0, o} < alo) alo = {8'd0, o};
                if (nhi > ahi)       ahi = nhi;
                off[id] = alo[31:0];
                len[id] = ahi - alo;
            end
        end
    endtask

    task add_read;
        input [7:0]  id;
        input [31:0] o;
        input [32:0] l;
        begin
            span(id, o, l);
            if (id < 8) use_r[id] = 1'b1;
        end
    endtask

    task add_write;
        input [7:0]  id;
        input [31:0] o;
        input [32:0] l;
        begin
            span(id, o, l);
            if (id < 8) use_w[id] = 1'b1;
        end
    endtask

    always @* begin
        kind         = K_NONE;
        opcode       = f_op;
        subop        = f_sub;
        job          = cmd[47:32];
        dep_job      = cmd[63:48];
        dep_slot     = cmd[111:104];
        ev_slot      = cmd[103:96];
        dep_en       = f_flags[`NPU_FLAG_DEP_EN];
        inplace      = f_flags[`NPU_FLAG_INPLACE];
        irq          = f_flags[`NPU_FLAG_IRQ];
        rel_target_slot = 8'd0;
        rel_target_job  = 16'd0;
        read_mask    = 8'd0;
        write_mask   = 8'd0;
        chk_status   = `NPU_ST_OK;
        ovf          = 1'b0;

        for (b = 0; b < 8; b = b + 1) begin
            use_r[b] = 1'b0;
            use_w[b] = 1'b0;
            off[b]   = 32'd0;
            len[b]   = 33'd0;
        end

        case (f_op)
        // ------------------------------------------------------------ LOAD
        `NPU_OP_LOAD: begin
            kind = K_DMA;
            if (f_sub == 8'd0) begin
                // local extent = (rows-1)*local_stride + row_bytes
                t64  = {32'd0, p4[15:0] - 16'd1} * {32'd0, p6};
                t64  = t64 + {32'd0, p3};
                if (t64 > 64'h0000_0000_FFFF_FFFF) ovf = 1'b1;
                add_write(f_dstb, p2, t64[32:0]);
            end
            // LOAD_W touches no scratchpad buffer (weight buffer is separate)
        end

        // ----------------------------------------------------------- STORE
        `NPU_OP_STORE: begin
            kind = K_DMA;
            t64  = {32'd0, p4[15:0] - 16'd1} * {32'd0, p6};
            t64  = t64 + {32'd0, p3};
            if (t64 > 64'h0000_0000_FFFF_FFFF) ovf = 1'b1;
            add_read(f_src0b, p2, t64[32:0]);
        end

        // ---------------------------------------------------------- MATMUL
        `NPU_OP_MATMUL: begin
            kind   = K_MPU;
            m_len  = {24'd0, p0[7:0]};
            kvalid = p2;
            nvalid = p3;

            // A: m_len rows of 16-bit elements, row stride P5
            t64 = {32'd0, m_len - 32'd1} * {32'd0, p5};
            t64 = t64 + ({32'd0, kvalid} << 1);
            if (t64 > 64'h0000_0000_FFFF_FFFF) ovf = 1'b1;
            add_read(f_src0b, p4, t64[32:0]);

            // B: when src1_buf is set the B operand is a w16 activation
            // frame in a scratchpad buffer.  Tile t owns groups
            // t*n_grp..t*n_grp+n_grp-1 of one flat frame, so the job spans
            // tiles*n_grp groups of stride P9, 32 B per k-step.
            if (cmd[79:72] != 8'd0) begin
                g8  = {24'd0, p0[15:8]};
                s4  = {16'd0, p1[31:16]};
                t64 = {32'd0, g8} * {16'd0, p1[15:0]};      // tiles * n_grp
                t64 = (t64 - 64'd1) * {32'd0, p9};
                t64 = t64 + ({32'd0, s4} << 5);
                if (t64 > 64'h0000_0000_FFFF_FFFF) ovf = 1'b1;
                add_read(cmd[79:72], p6, t64[32:0]);
            end

            // D: m_len rows of dst elements, row stride P8
            t64 = {32'd0, m_len - 32'd1} * {32'd0, p8};
            t64b = {32'd0, nvalid} * {32'd0, esd};
            if (t64b > 64'h0000_0000_FFFF_FFFF) ovf = 1'b1;
            t64 = t64 + t64b;
            if (t64 > 64'h0000_0000_FFFF_FFFF) ovf = 1'b1;
            add_write(f_dstb, p7, t64[32:0]);
        end

        // ---------------------------------------------------------- VECTOR
        `NPU_OP_VECTOR: begin
            kind  = K_VPU;
            count = p0;

            // src0 always covers count elements
            t64 = {32'd0, count} * {32'd0, es0};
            if (t64 > 64'h0000_0000_FFFF_FFFF) ovf = 1'b1;
            add_read(f_src0b, p1, t64[32:0]);

            // src1 extent depends on the subop and broadcast mode
            ecnt = {1'b0, count};
            case (f_sub[5:0])
            `NPU_VEC_ADD, `NPU_VEC_SUB, `NPU_VEC_MUL, `NPU_VEC_MAX: begin
                if (f_sub[7:6] == 2'b10) begin
                    ecnt = 33'd1;                       // scalar broadcast
                end else if (f_sub[7:6] == 2'b01) begin
                    if (count[0]) chk_status = `NPU_ST_BAD_FIELD;
                    ecnt = {1'b0, count} >> 1;          // pair broadcast
                end
                t64 = {31'd0, ecnt} * {32'd0, es1};
                if (t64 > 64'h0000_0000_FFFF_FFFF) ovf = 1'b1;
                add_read(f_src1b, p2, t64[32:0]);
            end
            `NPU_VEC_SELECT: begin
                t64 = {32'd0, es1};
                add_read(f_src1b, p2, t64[32:0]);
                // mask: one byte per element, in its own buffer
                add_read(p5[7:0], p6, {1'b0, count});
            end
            `NPU_VEC_COMPARE_GT: begin
                t64 = {32'd0, count} * {32'd0, es1};
                if (t64 > 64'h0000_0000_FFFF_FFFF) ovf = 1'b1;
                add_read(f_src1b, p2, t64[32:0]);
            end
            default: begin
                // CONVERT, MUL_SCALAR, ADD_SCALAR, ROTATE_PAIR: no src1 read
            end
            endcase

            // dst: one byte per element only for COMPARE_GT
            if (f_sub[5:0] == `NPU_VEC_COMPARE_GT)
                t64 = {32'd0, count};
            else
                t64 = {32'd0, count} * {32'd0, esd};
            if (t64 > 64'h0000_0000_FFFF_FFFF) ovf = 1'b1;
            add_write(f_dstb, p3, t64[32:0]);

            // INPLACE: the shared buffer is taken once, as an exclusive write
            if (inplace && (f_dstb < 8)) use_r[f_dstb] = 1'b0;
        end

        // ---------------------------------------------------------- REDUCE
        `NPU_OP_REDUCE: begin
            kind  = K_VPU;
            count = p0;
            t64 = {32'd0, count} * {32'd0, es0};
            if (t64 > 64'h0000_0000_FFFF_FFFF) ovf = 1'b1;
            add_read(f_src0b, p1, t64[32:0]);
            add_write(f_dstb, p3, 33'd4);
            if (f_sub[7]) begin
                add_read(p5[7:0], p6, {1'b0, count});
            end
        end

        // ------------------------------------------------------------- SFU
        `NPU_OP_SFU: begin
            kind  = K_SFU;
            count = p0;
            t64 = {32'd0, count} * {32'd0, es0};
            if (t64 > 64'h0000_0000_FFFF_FFFF) ovf = 1'b1;
            add_read(f_src0b, p1, t64[32:0]);

            t64 = {32'd0, count} * {32'd0, esd};
            if (t64 > 64'h0000_0000_FFFF_FFFF) ovf = 1'b1;
            if (f_sub == 8'd3) begin
                // SINCOS: sin at P3 and cos at P2, one contiguous dst extent
                dlo = (p2 < p3) ? p2 : p3;
                dhi = (p2 > p3) ? p2 : p3;
                t64b = t64 + {32'd0, dhi};
                add_write(f_dstb, dlo, t64b[32:0] - {32'd0, dlo});
            end else begin
                add_write(f_dstb, p3, t64[32:0]);
            end
        end

        // ---------------------------------------------------------- LAYOUT
        `NPU_OP_LAYOUT: begin
            kind = K_VPU;
            if (f_sub == `NPU_LAYOUT_TRANSPOSE) begin
                rows = p0;
                cols = p1;
                t64 = {32'd0, rows - 32'd1} * {32'd0, p4};
                t64 = t64 + ({32'd0, cols} * {32'd0, es0});
                if (t64 > 64'h0000_0000_FFFF_FFFF) ovf = 1'b1;
                add_read(f_src0b, p2, t64[32:0]);

                t64 = {32'd0, cols - 32'd1} * {32'd0, p5};
                t64 = t64 + ({32'd0, rows} * {32'd0, esd});
                if (t64 > 64'h0000_0000_FFFF_FFFF) ovf = 1'b1;
                add_write(f_dstb, p3, t64[32:0]);
            end else begin
                // PACK_W16: K_valid x N_valid source, packed frame destination
                g8  = {24'd0, p5[7:0]};
                s4  = (p0 + 32'd3) >> 2;
                t64 = {32'd0, p0 - 32'd1} * {32'd0, p4};
                t64 = t64 + ({32'd0, p1} << 1);
                if (t64 > 64'h0000_0000_FFFF_FFFF) ovf = 1'b1;
                add_read(f_src0b, p2, t64[32:0]);

                t64 = {32'd0, g8} * 32'd32;
                t64 = t64 * {32'd0, s4};
                if (t64 > 64'h0000_0000_FFFF_FFFF) ovf = 1'b1;
                add_write(f_dstb, p3, t64[32:0]);
            end
        end

        // ------------------------------------------------------------ FENCE
        `NPU_OP_FENCE: begin
            kind = K_FENCE;
        end

        // ---------------------------------------------------------- RELEASE
        `NPU_OP_RELEASE: begin
            kind = K_RELEASE;
            rel_target_slot = p0[7:0];
            rel_target_job  = p0[23:8];
        end

        default: begin
            kind = K_NONE;
            chk_status = `NPU_ST_BAD_OPCODE;
        end
        endcase

        // ------------------------------------------------------------------
        // buffer configuration checks
        // ------------------------------------------------------------------
        if (ovf)
            chk_status = `NPU_ST_RANGE;

        for (b = 0; b < 8; b = b + 1) begin
            endb = {8'd0, off[b]} + {7'd0, len[b]};
            if (use_r[b] || use_w[b]) begin
                if (!buf_enable[b])
                    chk_status = `NPU_ST_BAD_FIELD;
                else if (endb > {8'd0, buf_size_flat[32*b +: 32]})
                    chk_status = `NPU_ST_RANGE;
            end
        end

        // within-command aliasing: a read range and a write range that share a
        // physical bank must not overlap.  (Same logical buffer read+write is
        // only reachable through a static error; report it as well.)
        for (r = 0; r < 8; r = r + 1) begin
            if (use_r[r]) begin
                for (w = 0; w < 8; w = w + 1) begin
                    if (use_w[w]) begin
                        if (r == w) begin
                            chk_status = `NPU_ST_BAD_FIELD;
                        end else if ((buf_bank_flat[8*r +: 8] ==
                                      buf_bank_flat[8*w +: 8])) begin
                            // physical byte ranges
                            t64  = {32'd0, buf_base_flat[32*r +: 32]} * B;
                            t64  = t64 + {32'd0, off[r]};
                            t64b = t64 + {32'd0, len[r]};
                            t64c = {32'd0, buf_base_flat[32*w +: 32]} * B;
                            t64c = t64c + {32'd0, off[w]};
                            if ((t64 < t64c + {32'd0, len[w]}) &&
                                (t64c < t64b))
                                chk_status = `NPU_ST_BAD_FIELD;
                        end
                    end
                end
            end
        end

        // ------------------------------------------------------------------
        // exported masks
        // ------------------------------------------------------------------
        read_mask  = use_r;
        write_mask = use_w;
    end

endmodule

`default_nettype wire
