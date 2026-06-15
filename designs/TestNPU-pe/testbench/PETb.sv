`timescale 1ns/1ps

`define CLK_PERIOD 20
`define ASSIGNMENT_DELAY 5
`define FINISH_TIME 2000000

`ifndef NUM_TEST_VECTORS
`define NUM_TEST_VECTORS 16
`endif

module PETb;

  localparam VECTOR_WIDTH = 96;
  localparam DATA_WIDTH = 32;

  reg clk;
  reg rst_n;

  reg  [DATA_WIDTH-1:0] pe_psum_in;
  reg  [15:0]           pe_weight_in;
  reg                   pe_accept_w_in;
  reg  [15:0]           pe_input_in;
  reg                   pe_valid_in;
  reg                   pe_switch_in;
  reg                   pe_enabled;

  wire [DATA_WIDTH-1:0] pe_psum_out;
  wire [15:0]           pe_weight_out;
  wire [15:0]           pe_input_out;
  wire                  pe_valid_out;
  wire                  pe_switch_out;

  reg [VECTOR_WIDTH-1:0] test_vectors [`NUM_TEST_VECTORS-1:0];

  pe pe_inst (
      .clk            (clk),
      .rst_n          (rst_n),
      .pe_psum_in     (pe_psum_in),
      .pe_weight_in   (pe_weight_in),
      .pe_accept_w_in (pe_accept_w_in),
      .pe_input_in    (pe_input_in),
      .pe_valid_in    (pe_valid_in),
      .pe_switch_in   (pe_switch_in),
      .pe_enabled     (pe_enabled),
      .pe_psum_out    (pe_psum_out),
      .pe_weight_out  (pe_weight_out),
      .pe_input_out   (pe_input_out),
      .pe_valid_out   (pe_valid_out),
      .pe_switch_out  (pe_switch_out)
  );

  always #(`CLK_PERIOD/2) clk = ~clk;

  task reset_dut;
    begin
      rst_n          <= 1'b0;
      pe_enabled     <= 1'b1;
      pe_accept_w_in <= 1'b0;
      pe_switch_in   <= 1'b0;
      pe_valid_in    <= 1'b0;
      pe_weight_in   <= 16'h0000;
      pe_input_in    <= 16'h0000;
      pe_psum_in     <= 32'h00000000;
      repeat (4) @(posedge clk);
      rst_n <= #`ASSIGNMENT_DELAY 1'b1;
      repeat (2) @(posedge clk);
    end
  endtask

  task run_vector;
    input integer idx;
    reg [31:0] expected_fp32;
    reg [31:0] psum_in_fp32;
    reg [15:0] activation_fp16;
    reg [15:0] weight_fp16;
    begin
      expected_fp32  = test_vectors[idx][95:64];
      psum_in_fp32   = test_vectors[idx][63:32];
      activation_fp16 = test_vectors[idx][31:16];
      weight_fp16    = test_vectors[idx][15:0];

      // Load the PE foreground weight. Holding accept and switch together
      // matches the direct-load path in pe.sv.
      pe_accept_w_in <= #`ASSIGNMENT_DELAY 1'b1;
      pe_switch_in   <= #`ASSIGNMENT_DELAY 1'b1;
      pe_weight_in   <= #`ASSIGNMENT_DELAY weight_fp16;
      pe_valid_in    <= #`ASSIGNMENT_DELAY 1'b0;
      @(posedge clk);

      pe_accept_w_in <= #`ASSIGNMENT_DELAY 1'b0;
      pe_switch_in   <= #`ASSIGNMENT_DELAY 1'b0;
      @(posedge clk);

      // Apply one MAC transaction. pe_psum_out updates two cycles after
      // pe_valid_in for LATENCY=0: one explicit mult register, then add/write.
      pe_input_in <= #`ASSIGNMENT_DELAY activation_fp16;
      pe_psum_in  <= #`ASSIGNMENT_DELAY psum_in_fp32;
      pe_valid_in <= #`ASSIGNMENT_DELAY 1'b1;
      @(posedge clk);
      pe_valid_in <= #`ASSIGNMENT_DELAY 1'b0;
      @(posedge clk);
      #1;

      $display(
          "vector %0d: weight=%h activation=%h psum_in=%h got=%h expected=%h",
          idx, weight_fp16, activation_fp16, psum_in_fp32, pe_psum_out, expected_fp32
      );
      assert (pe_psum_out === expected_fp32)
        else begin
          $error(
              "PE mismatch at vector %0d: got %h, expected %h",
              idx, pe_psum_out, expected_fp32
          );
          $finish(2);
        end

      @(posedge clk);
    end
  endtask

  integer i;

  initial begin
    $readmemh("inputs/test_vectors.txt", test_vectors);

    clk = 1'b0;
    reset_dut();

    for (i = 0; i < `NUM_TEST_VECTORS; i = i + 1) begin
      run_vector(i);
    end

    $display("PETb PASS: %0d vectors", `NUM_TEST_VECTORS);
    $finish;
  end

  initial begin
    $vcdplusfile("dump.vcd");
    $vcdplusmemon();
    $vcdpluson(0, PETb);
    #(`FINISH_TIME);
    $error("PETb timed out");
    $finish(2);
  end

endmodule
