`default_nettype none
// SFU micro-sequencer skeleton: counts cycles per opcode.
module sfu_microseq (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [3:0]  op,
    output reg         busy,
    output reg         done,
    output reg  [3:0]  step
);
    reg [3:0] cycles;
    reg [3:0] count;

    always @* begin
        case (op)
            4'd0: cycles = 4'd2;  // EXP2
            4'd1: cycles = 4'd2;  // LOG2
            4'd2: cycles = 4'd3;  // RCP
            4'd3: cycles = 4'd4;  // RSQRT
            4'd4: cycles = 4'd4;  // SQRT
            4'd5: cycles = 4'd4;  // SIN
            4'd6: cycles = 4'd4;  // COS
            4'd7: cycles = 4'd6;  // TANH
            4'd8: cycles = 4'd7;  // SINCOS
            default: cycles = 4'd2;
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy  <= 1'b0;
            done  <= 1'b0;
            step  <= 4'd0;
            count <= 4'd0;
        end else begin
            done <= 1'b0;
            if (start && !busy) begin
                busy  <= 1'b1;
                count <= 4'd0;
                step  <= 4'd0;
            end else if (busy) begin
                if (count + 4'd1 >= cycles) begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    step <= 4'd0;
                end else begin
                    count <= count + 4'd1;
                    step  <= step + 4'd1;
                end
            end
        end
    end
endmodule
`default_nettype wire
