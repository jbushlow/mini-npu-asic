`timescale 1ns/1ps

`define CLK_PERIOD 20
`define FINISH_TIME 2000000

module CoreTb;

  localparam IRAM_ADDR_WIDTH = 12;
  localparam SPAD_DATA_WIDTH = 256;
  localparam DM_ADDR_WIDTH   = 27;
  localparam DM_DATA_WIDTH   = 256;
  localparam DMA_SLOT_COUNT  = 2;

  reg clk;
  reg rst_n;

  reg start;
  wire done;
  wire illegal_op_o;

  reg                       instr_write_en;
  reg [IRAM_ADDR_WIDTH-1:0] iram_addr;
  reg [               63:0] dma_iram_din;

  reg [               15:0] base_addr;
  reg                       dma_wr_en;
  reg [SPAD_DATA_WIDTH-1:0] dma_wr_data;
  reg [               15:0] dma_write_pointer;
  reg                       dma_rd_en;
  wire [SPAD_DATA_WIDTH-1:0] dma_rd_data;
  reg [               15:0] dma_read_pointer;

  reg [IRAM_ADDR_WIDTH-1:0] pool_base;
  reg [               31:0] program_id_csr;
  reg [              127:0] kernel_arg_csr;

  wire                       dm_req_valid;
  reg                        dm_req_ready;
  wire                       dm_req_we;
  wire [  DM_ADDR_WIDTH-1:0] dm_req_addr;
  wire [                7:0] dm_req_len;
  wire [  DM_DATA_WIDTH-1:0] dm_req_wdata;
  wire [DM_DATA_WIDTH/8-1:0] dm_req_wstrb;
  reg                        dm_rsp_valid;
  wire                       dm_rsp_ready;
  reg  [  DM_DATA_WIDTH-1:0] dm_rsp_data;
  reg                        dm_rsp_last;
  reg  [                1:0] dm_rsp_resp;

  wire       dmu_err_o;
  wire [7:0] dmu_err_slot_o;
  wire [1:0] dmu_err_cause_o;
  wire [1:0] dmu_err_resp_o;
  wire       dmu_rsp_seen_o;

  wire [31:0] beat_count_o;
  wire [31:0] overlap_stat_o;

  wire        tma_req;
  wire        tma_dir;
  wire [15:0] tma_dm_base;
  wire [14:0] tma_mt_base;
  wire [15:0] tma_len;
  reg         tma_done;

  wire [31:0] perf_cnt_cycles_o;
  wire [31:0] perf_cnt_instrs_o;
  wire [DMA_SLOT_COUNT-1:0] dma_slot_done_o;

  core core_inst (
      .clk              (clk),
      .rst_n            (rst_n),
      .start            (start),
      .done             (done),
      .illegal_op_o     (illegal_op_o),
      .instr_write_en   (instr_write_en),
      .iram_addr        (iram_addr),
      .dma_iram_din     (dma_iram_din),
      .base_addr        (base_addr),
      .dma_wr_en        (dma_wr_en),
      .dma_wr_data      (dma_wr_data),
      .dma_write_pointer(dma_write_pointer),
      .dma_rd_en        (dma_rd_en),
      .dma_rd_data      (dma_rd_data),
      .dma_read_pointer (dma_read_pointer),
      .pool_base        (pool_base),
      .program_id_csr   (program_id_csr),
      .kernel_arg_csr   (kernel_arg_csr),
      .dm_req_valid     (dm_req_valid),
      .dm_req_ready     (dm_req_ready),
      .dm_req_we        (dm_req_we),
      .dm_req_addr      (dm_req_addr),
      .dm_req_len       (dm_req_len),
      .dm_req_wdata     (dm_req_wdata),
      .dm_req_wstrb     (dm_req_wstrb),
      .dm_rsp_valid     (dm_rsp_valid),
      .dm_rsp_ready     (dm_rsp_ready),
      .dm_rsp_data      (dm_rsp_data),
      .dm_rsp_last      (dm_rsp_last),
      .dm_rsp_resp      (dm_rsp_resp),
      .dmu_err_o        (dmu_err_o),
      .dmu_err_slot_o   (dmu_err_slot_o),
      .dmu_err_cause_o  (dmu_err_cause_o),
      .dmu_err_resp_o   (dmu_err_resp_o),
      .dmu_rsp_seen_o   (dmu_rsp_seen_o),
      .beat_count_o     (beat_count_o),
      .overlap_stat_o   (overlap_stat_o),
      .tma_req          (tma_req),
      .tma_dir          (tma_dir),
      .tma_dm_base      (tma_dm_base),
      .tma_mt_base      (tma_mt_base),
      .tma_len          (tma_len),
      .tma_done         (tma_done),
      .perf_cnt_cycles_o(perf_cnt_cycles_o),
      .perf_cnt_instrs_o(perf_cnt_instrs_o),
      .dma_slot_done_o  (dma_slot_done_o)
  );

  always #(`CLK_PERIOD/2) clk = ~clk;

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    start = 1'b0;
    instr_write_en = 1'b0;
    iram_addr = '0;
    dma_iram_din = '0;
    base_addr = '0;
    dma_wr_en = 1'b0;
    dma_wr_data = '0;
    dma_write_pointer = '0;
    dma_rd_en = 1'b0;
    dma_read_pointer = '0;
    pool_base = '0;
    program_id_csr = '0;
    kernel_arg_csr = '0;
    dm_req_ready = 1'b1;
    dm_rsp_valid = 1'b0;
    dm_rsp_data = '0;
    dm_rsp_last = 1'b0;
    dm_rsp_resp = 2'b00;
    tma_done = 1'b0;

    repeat (6) @(posedge clk);
    rst_n <= 1'b1;
    repeat (20) @(posedge clk);

    if (illegal_op_o) begin
      $error("CoreTb idle smoke saw illegal_op_o");
      $finish(2);
    end

    $display("CoreTb PASS: reset/idle smoke completed");
    $finish;
  end

  initial begin
    if ($test$plusargs("ASIC_DUMP_VCD")) begin
      $dumpfile("outputs/run.vcd");
      $dumpvars(0, CoreTb);
    end
    #(`FINISH_TIME);
    $error("CoreTb timed out");
    $finish(2);
  end

endmodule
