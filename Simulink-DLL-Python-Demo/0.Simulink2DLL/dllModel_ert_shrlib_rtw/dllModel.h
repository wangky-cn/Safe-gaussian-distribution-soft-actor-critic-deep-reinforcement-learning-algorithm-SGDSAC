/*
 * File: dllModel.h
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

#ifndef RTW_HEADER_dllModel_h_
#define RTW_HEADER_dllModel_h_
#ifndef dllModel_COMMON_INCLUDES_
#define dllModel_COMMON_INCLUDES_
#include "rtwtypes.h"
#include "rtw_continuous.h"
#include "rtw_solver.h"
#endif                                 /* dllModel_COMMON_INCLUDES_ */

#include "dllModel_types.h"
#include <string.h>

/* Macros for accessing real-time model data structure */
#ifndef rtmGetErrorStatus
#define rtmGetErrorStatus(rtm)         ((rtm)->errorStatus)
#endif

#ifndef rtmSetErrorStatus
#define rtmSetErrorStatus(rtm, val)    ((rtm)->errorStatus = (val))
#endif

/* Real-time Model Data Structure */
struct tag_RTM_dllModel_T {
  const char_T * volatile errorStatus;

  /*
   * Timing:
   * The following substructure contains information regarding
   * the timing information for the model.
   */
  struct {
    uint32_T clockTick0;
    uint32_T clockTickH0;
  } Timing;
};

/*
 * Exported Global Signals
 *
 * Note: Exported global signals are block signals with an exported global
 * storage class designation.  Code generation will declare the memory for
 * these signals and export their symbols.
 *
 */
extern real_T APUPCMD;                 /* '<S1>/Gain1' */
extern real_T Step;                    /* '<S1>/Digital Clock' */

/*
 * Exported Global Parameters
 *
 * Note: Exported global parameters are tunable parameters with an exported
 * global storage class designation.  Code generation will declare the memory for
 * these parameters and exports their symbols.
 *
 */
extern real_T InputSignal;             /* Variable: InputSignal
                                        * Referenced by: '<Root>/Constant'
                                        */

/* Model entry point functions */
extern void dllModel_initialize(void);
extern void dllModel_step(void);
extern void dllModel_terminate(void);

/* Real-time Model object */
extern RT_MODEL_dllModel_T *const dllModel_M;

/*-
 * The generated code includes comments that allow you to trace directly
 * back to the appropriate location in the model.  The basic format
 * is <system>/block_name, where system is the system number (uniquely
 * assigned by Simulink) and block_name is the name of the block.
 *
 * Use the MATLAB hilite_system command to trace the generated code back
 * to the model.  For example,
 *
 * hilite_system('<S3>')    - opens system 3
 * hilite_system('<S3>/Kp') - opens and selects block Kp which resides in S3
 *
 * Here is the system hierarchy for this model
 *
 * '<Root>' : 'dllModel'
 * '<S1>'   : 'dllModel/Subsystem'
 */
#endif                                 /* RTW_HEADER_dllModel_h_ */

/*
 * File trailer for generated code.
 *
 * [EOF]
 */
