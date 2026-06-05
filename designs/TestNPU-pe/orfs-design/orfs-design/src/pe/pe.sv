`timescale 1ns / 1ps

module pe #(
    parameter DATA_WIDTH = 32,
    parameter LATENCY    = 0
) (
    input  wire                  clk,
    input  wire                  rst_n,

    input  wire [DATA_WIDTH-1:0] pe_psum_in,
    input  wire [15:0]           pe_weight_in,
    input  wire                  pe_accept_w_in,

    input  wire [15:0]           pe_input_in,
    input  wire                  pe_valid_in,
    input  wire                  pe_switch_in,
    input  wire                  pe_enabled,

    output reg  [DATA_WIDTH-1:0] pe_psum_out,
    output reg  [15:0]           pe_weight_out,

    output reg  [15:0]           pe_input_out,
    output reg                   pe_valid_out,
    output reg                   pe_switch_out
);

  wire [DATA_WIDTH-1:0] mult_out;
  reg  [DATA_WIDTH-1:0] mult_out_reg;
  reg                   mult_valid_reg;
  wire [DATA_WIDTH-1:0] mac_out;

  reg [15:0] weight_reg_active;
  reg [15:0] weight_reg_inactive;

  fp16_mul_fp32 u_fp16_mul (
      .a      (pe_input_in),
      .b      (weight_reg_active),
      .result (mult_out)
  );

  fp32_add #(
      .LATENCY     (LATENCY),
      .FORMAT_MODE (0),
      .INT_BITS    (16),
      .FRAC_BITS   (16),
      .WIDTH       (DATA_WIDTH)
  ) adder (
      .clk    (clk),
      .rst_n  (rst_n),
      .a      (mult_out_reg),
      .b      (pe_psum_in),
      .result (mac_out)
  );

  localparam SR_DEPTH = 2 * LATENCY;

  reg        valid_sr  [0:SR_DEPTH];
  reg        switch_sr [0:SR_DEPTH];
  reg [15:0] input_sr  [0:SR_DEPTH];

  always @* begin
    valid_sr[0]  = pe_valid_in;
    switch_sr[0] = pe_switch_in;
    input_sr[0]  = pe_input_in;
  end

  genvar i;
  generate
    if (SR_DEPTH > 0) begin : gen_sr_block
      for (i = 1; i <= SR_DEPTH; i = i + 1) begin : gen_sr
        always @(posedge clk or negedge rst_n) begin
          if (!rst_n) begin
            valid_sr[i]  <= 1'b0;
            switch_sr[i] <= 1'b0;
            input_sr[i]  <= 16'h0000;
          end else begin
            valid_sr[i]  <= valid_sr[i-1];
            switch_sr[i] <= switch_sr[i-1];
            input_sr[i]  <= input_sr[i-1];
          end
        end
      end
    end
  endgenerate

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      pe_input_out        <= 16'h0000;
      pe_psum_out         <= {DATA_WIDTH{1'b0}};
      pe_weight_out       <= 16'h0000;
      pe_valid_out        <= 1'b0;
      pe_switch_out       <= 1'b0;
      weight_reg_active   <= 16'h0000;
      weight_reg_inactive <= 16'h0000;
      mult_out_reg        <= {DATA_WIDTH{1'b0}};
      mult_valid_reg      <= 1'b0;

    end else if (!pe_enabled) begin
      pe_input_out        <= 16'h0000;
      pe_psum_out         <= {DATA_WIDTH{1'b0}};
      pe_weight_out       <= 16'h0000;
      pe_valid_out        <= 1'b0;
      pe_switch_out       <= 1'b0;
      weight_reg_active   <= 16'h0000;
      weight_reg_inactive <= 16'h0000;
      mult_out_reg        <= {DATA_WIDTH{1'b0}};
      mult_valid_reg      <= 1'b0;

    end else begin
      pe_valid_out  <= valid_sr[SR_DEPTH];
      pe_switch_out <= switch_sr[SR_DEPTH];

      mult_out_reg   <= mult_out;
      mult_valid_reg <= valid_sr[SR_DEPTH];

      if (pe_accept_w_in) begin
        weight_reg_inactive <= pe_weight_in;
        pe_weight_out       <= pe_weight_in;
      end else begin
        pe_weight_out <= 16'h0000;
      end

      if (pe_switch_in) begin
        if (pe_accept_w_in) begin
          weight_reg_active <= pe_weight_in;
        end else begin
          weight_reg_active <= weight_reg_inactive;
        end
      end

      if (valid_sr[SR_DEPTH]) begin
        pe_input_out <= input_sr[SR_DEPTH];
      end

      if (mult_valid_reg) begin
        pe_psum_out <= mac_out;
      end else begin
        pe_psum_out <= {DATA_WIDTH{1'b0}};
      end
    end
  end

endmodule