// fp16_mul_fp32 — FP16 × FP16 → FP32 multiply.
//
// Yosys/ORFS-friendly version:
// - no logic
// - no always_comb
// - result declared as reg
// - all internal combinational signals declared as wire/reg
// - attributes kept in standard Verilog attribute syntax
//
// Special cases: NaN propagation, Inf×0→NaN, Inf, zero.
// Subnormal FP16 inputs are flushed to zero.

`timescale 1ns / 1ps

module fp16_mul_fp32 (
    input  wire [15:0] a,
    input  wire [15:0] b,
    output reg  [31:0] result
);

  // ----------------------------------------------------------------------
  // FP16 fields
  // ----------------------------------------------------------------------

  wire        sign_a;
  wire        sign_b;
  wire        sign_r;
  wire [7:0]  exp_a;
  wire [7:0]  exp_b;
  wire [10:0] mant_a;
  wire [10:0] mant_b;

  assign sign_a = a[15];
  assign sign_b = b[15];
  assign sign_r = sign_a ^ sign_b;

  // Zero-extend FP16 exponents to 8 bits for FP32 exponent arithmetic.
  assign exp_a = {3'b000, a[14:10]};
  assign exp_b = {3'b000, b[14:10]};

  // Hidden-1 prepended for normal inputs; hidden-0 for subnormals.
  // Subnormals are later flushed to zero.
  assign mant_a = (exp_a != 8'h00) ? {1'b1, a[9:0]} : {1'b0, a[9:0]};
  assign mant_b = (exp_b != 8'h00) ? {1'b1, b[9:0]} : {1'b0, b[9:0]};

  // ----------------------------------------------------------------------
  // Mantissa multiply
  // ----------------------------------------------------------------------

  // Attribute is harmless for Yosys/OpenROAD ASIC flow, and useful if this
  // module is ever synthesized by Vivado.
  (* use_dsp = "yes" *) wire [21:0] mant_prod;

  assign mant_prod = mant_a * mant_b;

  // ----------------------------------------------------------------------
  // Special-case flags
  // ----------------------------------------------------------------------

  wire zero_a;
  wire zero_b;
  wire inf_a;
  wire inf_b;
  wire nan_a;
  wire nan_b;

  assign zero_a = (exp_a == 8'h00) && (a[9:0] == 10'h000);
  assign zero_b = (exp_b == 8'h00) && (b[9:0] == 10'h000);

  assign inf_a  = (exp_a == 8'h1F) && (a[9:0] == 10'h000);
  assign inf_b  = (exp_b == 8'h1F) && (b[9:0] == 10'h000);

  assign nan_a  = (exp_a == 8'h1F) && (a[9:0] != 10'h000);
  assign nan_b  = (exp_b == 8'h1F) && (b[9:0] != 10'h000);

  // ----------------------------------------------------------------------
  // Result generation
  // ----------------------------------------------------------------------

  reg [7:0] exp_sum;

  always @* begin
    result  = 32'h00000000;
    exp_sum = 8'h00;

    if (nan_a || nan_b) begin
      result = 32'h7FC00000;

    end else if ((inf_a && zero_b) || (inf_b && zero_a)) begin
      // Inf × 0 → NaN
      result = 32'h7FC00000;

    end else if (inf_a || inf_b) begin
      result = {sign_r, 8'hFF, 23'h000000};

    end else if (zero_a || zero_b || (exp_a == 8'h00) || (exp_b == 8'h00)) begin
      // Zero or subnormal input → flush result to signed zero.
      result = {sign_r, 8'h00, 23'h000000};

    end else begin
      if (mant_prod[21]) begin
        // Product is in [2.0, 4.0), so exponent gets extra +1.
        //
        // FP32 exponent:
        //   (exp_a - 15) + (exp_b - 15) + 127 + 1
        // = exp_a + exp_b + 98
        exp_sum = exp_a + exp_b + 8'd98;

        if (exp_sum >= 8'hFF) begin
          result = {sign_r, 8'hFF, 23'h000000};
        end else begin
          result = {sign_r, exp_sum, mant_prod[20:0], 2'b00};
        end

      end else begin
        // Product is in [1.0, 2.0).
        //
        // FP32 exponent:
        //   (exp_a - 15) + (exp_b - 15) + 127
        // = exp_a + exp_b + 97
        exp_sum = exp_a + exp_b + 8'd97;

        if (exp_sum >= 8'hFF) begin
          result = {sign_r, 8'hFF, 23'h000000};
        end else begin
          result = {sign_r, exp_sum, mant_prod[19:0], 3'b000};
        end
      end
    end
  end

endmodule