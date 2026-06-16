// fp32_add — IEEE 754 FP32 add.
//
// Yosys/ORFS-friendly version:
// - no logic
// - no always_comb / always_ff
// - no int unsigned parameters
// - no string FORMAT parameter comparison
// - no DPI-C path
// - no SystemVerilog casts like int'(...)
//
// FORMAT_MODE:
//   0 = FP32
//   1 = fixed-point saturating integer add
//
// LATENCY:
//   0 = combinational FP32 add
//   1 = one pipeline register after mantissa add/alignment
//
// Note:
// - This is still not a fully rounded IEEE-754 adder. It preserves the
//   original behavior: alignment, add/subtract, normalize, truncate.

`timescale 1ns / 1ps

module fp32_add #(
    parameter LATENCY     = 0,
    parameter FORMAT_MODE = 0,   // 0 = FP32, 1 = fixed-point
    parameter INT_BITS    = 16,
    parameter FRAC_BITS   = 16,
    parameter WIDTH       = 32
) (
    input  wire             clk,
    input  wire             rst_n,
    input  wire [WIDTH-1:0] a,
    input  wire [WIDTH-1:0] b,
    output reg  [WIDTH-1:0] result
);

  generate
    if (FORMAT_MODE == 0) begin : fp32_mode

      // ------------------------------------------------------------------
      // Input extraction
      // ------------------------------------------------------------------

      wire        a_sign;
      wire        b_sign;
      wire [7:0]  a_exp;
      wire [7:0]  b_exp;
      wire [22:0] a_mant;
      wire [22:0] b_mant;

      wire a_nan;
      wire b_nan;
      wire a_inf;
      wire b_inf;
      wire a_zero;
      wire b_zero;

      assign a_sign = a[31];
      assign b_sign = b[31];
      assign a_exp  = a[30:23];
      assign b_exp  = b[30:23];
      assign a_mant = a[22:0];
      assign b_mant = b[22:0];

      assign a_nan  = (a_exp == 8'hFF) && (a_mant != 23'h000000);
      assign b_nan  = (b_exp == 8'hFF) && (b_mant != 23'h000000);

      assign a_inf  = (a_exp == 8'hFF) && (a_mant == 23'h000000);
      assign b_inf  = (b_exp == 8'hFF) && (b_mant == 23'h000000);

      assign a_zero = (a_exp == 8'h00) && (a_mant == 23'h000000);
      assign b_zero = (b_exp == 8'h00) && (b_mant == 23'h000000);

      if (LATENCY == 0) begin : lat0

        // --------------------------------------------------------------
        // Purely combinational FP32 add
        // --------------------------------------------------------------

        reg        result_sign;
        reg [23:0] a_mant_ext;
        reg [23:0] b_mant_ext;
        reg [24:0] sum_mant;
        reg [24:0] mant_sub;

        integer larger_exp_i;
        integer exp_diff_i;
        integer result_exp_i;
        integer shift;
        integer i;

        reg normalize_done;

        always @* begin
          result         = {WIDTH{1'b0}};
          result_sign    = 1'b0;
          a_mant_ext     = 24'h000000;
          b_mant_ext     = 24'h000000;
          sum_mant       = 25'h0000000;
          mant_sub       = 25'h0000000;
          larger_exp_i   = 0;
          exp_diff_i     = 0;
          result_exp_i   = 0;
          shift          = 0;
          normalize_done = 1'b0;

          if (a_nan || b_nan) begin
            result = 32'h7FC00000;

          end else if (a_inf && b_inf) begin
            if (a_sign == b_sign)
              result = {a_sign, 8'hFF, 23'h000000};
            else
              result = 32'h7FC00000;

          end else if (a_inf) begin
            result = {a_sign, 8'hFF, 23'h000000};

          end else if (b_inf) begin
            result = {b_sign, 8'hFF, 23'h000000};

          end else if (a_zero && b_zero) begin
            result = {a_sign & b_sign, 8'h00, 23'h000000};

          end else if (a_zero) begin
            result = b;

          end else if (b_zero) begin
            result = a;

          end else begin
            if (a_exp == 8'h00)
              a_mant_ext = {1'b0, a_mant};
            else
              a_mant_ext = {1'b1, a_mant};

            if (b_exp == 8'h00)
              b_mant_ext = {1'b0, b_mant};
            else
              b_mant_ext = {1'b1, b_mant};

            if (a_exp >= b_exp) begin
              larger_exp_i = a_exp;
              exp_diff_i   = a_exp - b_exp;
              b_mant_ext   = b_mant_ext >> exp_diff_i;
            end else begin
              larger_exp_i = b_exp;
              exp_diff_i   = b_exp - a_exp;
              a_mant_ext   = a_mant_ext >> exp_diff_i;
            end

            if (a_sign == b_sign) begin
              sum_mant    = a_mant_ext + b_mant_ext;
              result_sign = a_sign;
            end else begin
              if (a_mant_ext >= b_mant_ext) begin
                sum_mant    = a_mant_ext - b_mant_ext;
                result_sign = a_sign;
              end else begin
                sum_mant    = b_mant_ext - a_mant_ext;
                result_sign = b_sign;
              end
            end

            if (larger_exp_i == 0) begin
              if (sum_mant == 25'h0000000) begin
                result = 32'h00000000;
              end else if (sum_mant[23]) begin
                result = {result_sign, 8'h01, sum_mant[22:0]};
              end else begin
                result = {result_sign, 8'h00, sum_mant[22:0]};
              end

            end else begin
              result_exp_i   = larger_exp_i;
              normalize_done = 1'b0;

              if (sum_mant[24]) begin
                sum_mant       = sum_mant >> 1;
                result_exp_i   = result_exp_i + 1;
                normalize_done = 1'b1;
              end else if (sum_mant[23]) begin
                normalize_done = 1'b1;
              end

              if (!normalize_done) begin
                for (i = 22; i >= 0; i = i - 1) begin
                  if (sum_mant[i] && !normalize_done) begin
                    sum_mant       = sum_mant << (23 - i);
                    result_exp_i   = result_exp_i - (23 - i);
                    normalize_done = 1'b1;
                  end
                end
              end

              if (sum_mant == 25'h0000000) begin
                result = 32'h00000000;

              end else if (result_exp_i >= 255) begin
                result = {result_sign, 8'hFF, 23'h000000};

              end else if (result_exp_i <= 0) begin
                shift = 1 - result_exp_i;

                if (shift >= 25) begin
                  result = {result_sign, 8'h00, 23'h000000};
                end else begin
                  mant_sub = sum_mant >> shift;
                  result   = {result_sign, 8'h00, mant_sub[22:0]};
                end

              end else begin
                result = {result_sign, result_exp_i[7:0], sum_mant[22:0]};
              end
            end
          end
        end

      end else begin : lat1

        // --------------------------------------------------------------
        // Two-stage version
        // --------------------------------------------------------------

        // Stage 1 combinational outputs.
        reg        s1_is_special;
        reg [31:0] s1_special_result;
        reg        s1_result_sign;
        reg [7:0]  s1_larger_exp;
        reg [24:0] s1_sum_mant;

        reg [23:0] s1_a_mant_ext;
        reg [23:0] s1_b_mant_ext;
        integer    s1_exp_diff;

        // Pipeline registers.
        reg        p1_is_special;
        reg [31:0] p1_special_result;
        reg        p1_result_sign;
        reg [7:0]  p1_larger_exp;
        reg [24:0] p1_sum_mant;

        // Stage 2 temporaries.
        reg        s2_normalize_done;
        integer    s2_result_exp;
        integer    s2_shift;
        integer    s2_i;
        reg [24:0] s2_sum_mant;
        reg [24:0] s2_mant_sub;

        // Stage 1: special cases, alignment, mantissa add/subtract.
        always @* begin
          s1_is_special     = 1'b0;
          s1_special_result = 32'h00000000;
          s1_result_sign    = 1'b0;
          s1_larger_exp     = 8'h00;
          s1_sum_mant       = 25'h0000000;
          s1_a_mant_ext     = 24'h000000;
          s1_b_mant_ext     = 24'h000000;
          s1_exp_diff       = 0;

          if (a_nan || b_nan) begin
            s1_is_special     = 1'b1;
            s1_special_result = 32'h7FC00000;

          end else if (a_inf && b_inf) begin
            s1_is_special = 1'b1;
            if (a_sign == b_sign)
              s1_special_result = {a_sign, 8'hFF, 23'h000000};
            else
              s1_special_result = 32'h7FC00000;

          end else if (a_inf) begin
            s1_is_special     = 1'b1;
            s1_special_result = {a_sign, 8'hFF, 23'h000000};

          end else if (b_inf) begin
            s1_is_special     = 1'b1;
            s1_special_result = {b_sign, 8'hFF, 23'h000000};

          end else if (a_zero && b_zero) begin
            s1_is_special     = 1'b1;
            s1_special_result = {a_sign & b_sign, 8'h00, 23'h000000};

          end else if (a_zero) begin
            s1_is_special     = 1'b1;
            s1_special_result = b;

          end else if (b_zero) begin
            s1_is_special     = 1'b1;
            s1_special_result = a;

          end else begin
            if (a_exp == 8'h00)
              s1_a_mant_ext = {1'b0, a_mant};
            else
              s1_a_mant_ext = {1'b1, a_mant};

            if (b_exp == 8'h00)
              s1_b_mant_ext = {1'b0, b_mant};
            else
              s1_b_mant_ext = {1'b1, b_mant};

            if (a_exp >= b_exp) begin
              s1_larger_exp = a_exp;
              s1_exp_diff   = a_exp - b_exp;
              s1_b_mant_ext = s1_b_mant_ext >> s1_exp_diff;
            end else begin
              s1_larger_exp = b_exp;
              s1_exp_diff   = b_exp - a_exp;
              s1_a_mant_ext = s1_a_mant_ext >> s1_exp_diff;
            end

            if (a_sign == b_sign) begin
              s1_sum_mant    = s1_a_mant_ext + s1_b_mant_ext;
              s1_result_sign = a_sign;
            end else begin
              if (s1_a_mant_ext >= s1_b_mant_ext) begin
                s1_sum_mant    = s1_a_mant_ext - s1_b_mant_ext;
                s1_result_sign = a_sign;
              end else begin
                s1_sum_mant    = s1_b_mant_ext - s1_a_mant_ext;
                s1_result_sign = b_sign;
              end
            end
          end
        end

        // Pipeline FF.
        // rst_n was unused in your original code. Here I use it, which is
        // usually better for ASIC synthesis and avoids an unused reset.
        always @(posedge clk or negedge rst_n) begin
          if (!rst_n) begin
            p1_is_special     <= 1'b0;
            p1_special_result <= 32'h00000000;
            p1_result_sign    <= 1'b0;
            p1_larger_exp     <= 8'h00;
            p1_sum_mant       <= 25'h0000000;
          end else begin
            p1_is_special     <= s1_is_special;
            p1_special_result <= s1_special_result;
            p1_result_sign    <= s1_result_sign;
            p1_larger_exp     <= s1_larger_exp;
            p1_sum_mant       <= s1_sum_mant;
          end
        end

        // Stage 2: normalization.
        always @* begin
          result              = {WIDTH{1'b0}};
          s2_result_exp       = 0;
          s2_normalize_done   = 1'b0;
          s2_shift            = 0;
          s2_sum_mant         = 25'h0000000;
          s2_mant_sub         = 25'h0000000;
          s2_i                = 0;

          if (p1_is_special) begin
            result = p1_special_result;

          end else begin
            s2_sum_mant = p1_sum_mant;

            if (p1_larger_exp == 8'h00) begin
              if (s2_sum_mant == 25'h0000000) begin
                result = 32'h00000000;
              end else if (s2_sum_mant[23]) begin
                result = {p1_result_sign, 8'h01, s2_sum_mant[22:0]};
              end else begin
                result = {p1_result_sign, 8'h00, s2_sum_mant[22:0]};
              end

            end else begin
              s2_result_exp     = p1_larger_exp;
              s2_normalize_done = 1'b0;

              if (s2_sum_mant[24]) begin
                s2_sum_mant       = s2_sum_mant >> 1;
                s2_result_exp     = s2_result_exp + 1;
                s2_normalize_done = 1'b1;
              end else if (s2_sum_mant[23]) begin
                s2_normalize_done = 1'b1;
              end

              if (!s2_normalize_done) begin
                for (s2_i = 22; s2_i >= 0; s2_i = s2_i - 1) begin
                  if (s2_sum_mant[s2_i] && !s2_normalize_done) begin
                    s2_sum_mant       = s2_sum_mant << (23 - s2_i);
                    s2_result_exp     = s2_result_exp - (23 - s2_i);
                    s2_normalize_done = 1'b1;
                  end
                end
              end

              if (s2_sum_mant == 25'h0000000) begin
                result = 32'h00000000;

              end else if (s2_result_exp >= 255) begin
                result = {p1_result_sign, 8'hFF, 23'h000000};

              end else if (s2_result_exp <= 0) begin
                s2_shift = 1 - s2_result_exp;

                if (s2_shift >= 25) begin
                  result = {p1_result_sign, 8'h00, 23'h000000};
                end else begin
                  s2_mant_sub = s2_sum_mant >> s2_shift;
                  result      = {p1_result_sign, 8'h00, s2_mant_sub[22:0]};
                end

              end else begin
                result = {p1_result_sign, s2_result_exp[7:0], s2_sum_mant[22:0]};
              end
            end
          end
        end

      end

    end else begin : fixed_point_mode

      // ------------------------------------------------------------------
      // Fixed-point / integer saturating adder
      // ------------------------------------------------------------------

      wire [WIDTH-1:0] sum;
      wire             overflow;

      assign sum = a + b;

      assign overflow =
          (a[WIDTH-1] == b[WIDTH-1]) &&
          (sum[WIDTH-1] != a[WIDTH-1]);

      always @* begin
        if (overflow) begin
          if (a[WIDTH-1])
            result = {1'b1, {WIDTH-1{1'b0}}};
          else
            result = {1'b0, {WIDTH-1{1'b1}}};
        end else begin
          result = sum;
        end
      end

    end
  endgenerate

endmodule