% Running the main.m automatically generates the DLL.

bdclose('all');
clear;
model = 'dllModel';
feval([model '_RAM_ROM']);
model = 'dllModel';
open_system(model);
model = 'dllModel';
rtwbuild(model);