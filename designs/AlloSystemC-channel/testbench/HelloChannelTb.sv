`timescale 1ns/1ps

`define CLK_PERIOD 10
`define FINISH_TIME 10000

module HelloChannelTb;

  localparam integer NUM_VECTORS = 8;

  reg         clk;
  reg         rst;
  wire        done;
  reg         v8_vld;
  wire        v8_rdy;
  reg  [31:0] v8_dat;
  wire        v9_vld;
  reg         v9_rdy;
  wire [31:0] v9_dat;

  reg [31:0] vectors [0:NUM_VECTORS-1];
  integer sent;
  integer received;
  integer cycles;

  hello_channel dut (
      .clk   (clk),
      .rst   (rst),
      .done  (done),
      .v8_vld(v8_vld),
      .v8_rdy(v8_rdy),
      .v8_dat(v8_dat),
      .v9_vld(v9_vld),
      .v9_rdy(v9_rdy),
      .v9_dat(v9_dat)
  );

  always #(`CLK_PERIOD/2) clk = ~clk;

  initial begin
    vectors[0] = 32'h00000000;
    vectors[1] = 32'h00000001;
    vectors[2] = 32'hffffffff;
    vectors[3] = 32'h80000000;
    vectors[4] = 32'h7fffffff;
    vectors[5] = 32'hdeadbeef;
    vectors[6] = 32'h12345678;
    vectors[7] = 32'ha5a55a5a;

    clk      = 1'b0;
    rst      = 1'b0;
    v8_vld   = 1'b0;
    v8_dat   = 32'b0;
    v9_rdy   = 1'b0;
    sent     = 0;
    received = 0;
    cycles   = 0;

    repeat (5) @(posedge clk);
    rst <= 1'b1;
  end

  // Hold each input stable until the DUT accepts it.
  always @(posedge clk) begin
    if (!rst) begin
      v8_vld <= 1'b0;
      v8_dat <= 32'b0;
      sent   <= 0;
    end else begin
      if (v8_vld && v8_rdy) begin
        sent <= sent + 1;
        if (sent + 1 < NUM_VECTORS) begin
          v8_dat <= vectors[sent + 1];
        end else begin
          v8_vld <= 1'b0;
        end
      end else if (!v8_vld && sent < NUM_VECTORS) begin
        v8_vld <= 1'b1;
        v8_dat <= vectors[sent];
      end
    end
  end

  // Deterministic backpressure: stall two cycles out of every five.
  always @(posedge clk) begin
    if (!rst) begin
      cycles <= 0;
      v9_rdy <= 1'b0;
    end else begin
      cycles <= cycles + 1;
      v9_rdy <= ((cycles % 5) != 1) && ((cycles % 5) != 2);
    end
  end

  always @(posedge clk) begin
    if (rst && v9_vld && v9_rdy) begin
      if (received >= NUM_VECTORS) begin
        $error("Unexpected extra output 0x%08x", v9_dat);
        $finish(2);
      end
      if (v9_dat !== vectors[received]) begin
        $error("Output %0d mismatch: expected 0x%08x, got 0x%08x",
               received, vectors[received], v9_dat);
        $finish(2);
      end
      received <= received + 1;
    end
  end

  always @(posedge clk) begin
    if (rst && received == NUM_VECTORS && done) begin
      $display("HelloChannelTb PASS: checked %0d ready/valid transfers", received);
      $finish;
    end
  end

  initial begin
    $vcdplusfile("dump.vcd");
    $vcdplusmemon();
    $vcdpluson(0, HelloChannelTb);
    #(`FINISH_TIME);
    $error("HelloChannelTb timed out (sent=%0d received=%0d done=%b)",
           sent, received, done);
    $finish(2);
  end

endmodule
