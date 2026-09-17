// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// VX_tm_pkg — TaskManager shared encodings. One source of truth for the owner /
// phase / role / src codes the TSM, PQM, PR and top all agree on (hw_spec §1).
// The model-side 13-state owner enum (warpspec_task_owner_state) is authoritative
// for semantics; these are the 7 RTL encodings plus the reserved gap.

`include "VX_define.vh"

package VX_tm_pkg;

    // owner of a task (3 bits): exactly one of these holds a live task at a time
    localparam logic [2:0] TM_OWN_FREE     = 3'd0;
    localparam logic [2:0] TM_OWN_INIT     = 3'd1;  // init warp holds it (genTask)
    localparam logic [2:0] TM_OWN_TRACE_WK = 3'd2;  // TRACE worker
    localparam logic [2:0] TM_OWN_RT_SLOT  = 3'd3;  // RT Unit RaySlot (traversing)
    localparam logic [2:0] TM_OWN_SHADE_WK = 3'd4;  // SHADE worker
    localparam logic [2:0] TM_OWN_FINAL_WK = 3'd5;  // FINALIZE worker
    localparam logic [2:0] TM_OWN_R_COMMIT = 3'd6;  // Phase Router commit state

    // phase / queue a task is published to (2 bits). The 4th code (reserved in
    // hw_spec §2.4) is the recycle: a FINALIZE worker publishes RELEASE to retire
    // the task -- owner->FREE, the id returns to the Free-ID FIFO, live_count--.
    localparam logic [1:0] TM_PH_TRACE_READY    = 2'd0;
    localparam logic [1:0] TM_PH_SHADE_READY    = 2'd1;
    localparam logic [1:0] TM_PH_FINALIZE_READY = 2'd2;
    localparam logic [1:0] TM_PH_RELEASE        = 2'd3;

    // worker role requested by getWork (2 bits)
    localparam logic [1:0] TM_ROLE_TRACE    = 2'd0;
    localparam logic [1:0] TM_ROLE_SHADE    = 2'd1;
    localparam logic [1:0] TM_ROLE_FINALIZE = 2'd2;

    // publishPhase source (2 bits)
    localparam logic [1:0] TM_SRC_INIT   = 2'd0;
    localparam logic [1:0] TM_SRC_WORKER = 2'd1;

    // map a getWork role to the owner code its packet migrates to
    function automatic logic [2:0] tm_role_owner(input logic [1:0] role);
        case (role)
            TM_ROLE_TRACE:    tm_role_owner = TM_OWN_TRACE_WK;
            TM_ROLE_SHADE:    tm_role_owner = TM_OWN_SHADE_WK;
            TM_ROLE_FINALIZE: tm_role_owner = TM_OWN_FINAL_WK;
            default:          tm_role_owner = TM_OWN_FREE;
        endcase
    endfunction

endpackage
