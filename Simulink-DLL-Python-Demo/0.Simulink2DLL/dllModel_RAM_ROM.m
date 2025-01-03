%%
APUPCMD = mpt.Signal;
APUPCMD.CoderInfo.StorageClass = 'ExportedGlobal';
APUPCMD.DataType = 'double';
APUPCMD.InitialValue = '0';
APUPCMD.Dimensions = 1;

Step = mpt.Signal;
Step.CoderInfo.StorageClass = 'ExportedGlobal';
Step.DataType = 'double';
Step.InitialValue = '0';
Step.Dimensions = 1;

%%
InputSignal=Simulink.Parameter;
InputSignal.Value=0;
InputSignal.CoderInfo.StorageClass='ExportedGlobal';




