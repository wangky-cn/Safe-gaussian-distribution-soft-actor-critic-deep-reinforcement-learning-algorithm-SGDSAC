/*
 * File: dllModel.c
 *
 * Code generated for Simulink model 'dllModel'.
 *
 * Model version                  : 9.14
 * Simulink Coder version         : 9.8 (R2022b) 13-May-2022
 * C/C++ source code generated on : Fri Jan  3 10:12:47 2025
 *
 * Target selection: ert_shrlib.tlc
 * Embedded hardware selection: Intel->x86-64 (Windows64)
 * Code generation objectives: Unspecified
 * Validation result: Not run
 */

#include "dllModel.h"
#include "rtwtypes.h"

/* Exported block signals */
real_T APUPCMD;                        /* '<S1>/Gain1' */
real_T Step;                           /* '<S1>/Digital Clock' */

/* Exported block parameters */
real_T InputSignal = 0.0;              /* Variable: InputSignal
                                        * Referenced by: '<Root>/Constant'
                                        */

/* Real-time model */
static RT_MODEL_dllModel_T dllModel_M_;
RT_MODEL_dllModel_T *const dllModel_M = &dllModel_M_;

/* Model step function */
void dllModel_step(void)
{
  /* Gain: '<S1>/Gain1' incorporates:
   *  Constant: '<Root>/Constant'
   */
  APUPCMD = 2.0 * InputSignal;

  /* DigitalClock: '<S1>/Digital Clock' */
  Step = (((dllModel_M->Timing.clockTick0+dllModel_M->Timing.clockTickH0*
            4294967296.0)) * 0.001);

  /* Update absolute time for base rate */
  /* The "clockTick0" counts the number of times the code of this task has
   * been executed. The resolution of this integer timer is 0.001, which is the step size
   * of the task. Size of "clockTick0" ensures timer will not overflow during the
   * application lifespan selected.
   * Timer of this task consists of two 32 bit unsigned integers.
   * The two integers represent the low bits Timing.clockTick0 and the high bits
   * Timing.clockTickH0. When the low bit overflows to 0, the high bits increment.
   */
  dllModel_M->Timing.clockTick0++;
  if (!dllModel_M->Timing.clockTick0) {
    dllModel_M->Timing.clockTickH0++;
  }
}

/* Model initialize function */
void dllModel_initialize(void)
{
  /* Registration code */

  /* initialize real-time model */
  (void) memset((void *)dllModel_M, 0,
                sizeof(RT_MODEL_dllModel_T));

  /* block I/O */

  /* exported global signals */
  APUPCMD = 0.0;
  Step = 0.0;
}

/* Model terminate function */
void dllModel_terminate(void)
{
  /* (no terminate code required) */
}

/*
 * File trailer for generated code.
 *
 * [EOF]
 */
