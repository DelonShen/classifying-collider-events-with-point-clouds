��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%��L>"
Ttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718��

�
&dnn__flatten/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*7
shared_name(&dnn__flatten/batch_normalization/gamma
�
:dnn__flatten/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp&dnn__flatten/batch_normalization/gamma*
_output_shapes
:i*
dtype0
�
%dnn__flatten/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*6
shared_name'%dnn__flatten/batch_normalization/beta
�
9dnn__flatten/batch_normalization/beta/Read/ReadVariableOpReadVariableOp%dnn__flatten/batch_normalization/beta*
_output_shapes
:i*
dtype0
�
,dnn__flatten/batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*=
shared_name.,dnn__flatten/batch_normalization/moving_mean
�
@dnn__flatten/batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOp,dnn__flatten/batch_normalization/moving_mean*
_output_shapes
:i*
dtype0
�
0dnn__flatten/batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*A
shared_name20dnn__flatten/batch_normalization/moving_variance
�
Ddnn__flatten/batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp0dnn__flatten/batch_normalization/moving_variance*
_output_shapes
:i*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
�
dnn__flatten/dense_92/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	i�*-
shared_namednn__flatten/dense_92/kernel
�
0dnn__flatten/dense_92/kernel/Read/ReadVariableOpReadVariableOpdnn__flatten/dense_92/kernel*
_output_shapes
:	i�*
dtype0
�
dnn__flatten/dense_92/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namednn__flatten/dense_92/bias
�
.dnn__flatten/dense_92/bias/Read/ReadVariableOpReadVariableOpdnn__flatten/dense_92/bias*
_output_shapes	
:�*
dtype0
�
dnn__flatten/dense_93/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*-
shared_namednn__flatten/dense_93/kernel
�
0dnn__flatten/dense_93/kernel/Read/ReadVariableOpReadVariableOpdnn__flatten/dense_93/kernel* 
_output_shapes
:
��*
dtype0
�
dnn__flatten/dense_93/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namednn__flatten/dense_93/bias
�
.dnn__flatten/dense_93/bias/Read/ReadVariableOpReadVariableOpdnn__flatten/dense_93/bias*
_output_shapes	
:�*
dtype0
�
dnn__flatten/dense_94/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*-
shared_namednn__flatten/dense_94/kernel
�
0dnn__flatten/dense_94/kernel/Read/ReadVariableOpReadVariableOpdnn__flatten/dense_94/kernel* 
_output_shapes
:
��*
dtype0
�
dnn__flatten/dense_94/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namednn__flatten/dense_94/bias
�
.dnn__flatten/dense_94/bias/Read/ReadVariableOpReadVariableOpdnn__flatten/dense_94/bias*
_output_shapes	
:�*
dtype0
�
dnn__flatten/dense_95/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*-
shared_namednn__flatten/dense_95/kernel
�
0dnn__flatten/dense_95/kernel/Read/ReadVariableOpReadVariableOpdnn__flatten/dense_95/kernel*
_output_shapes
:	�*
dtype0
�
dnn__flatten/dense_95/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namednn__flatten/dense_95/bias
�
.dnn__flatten/dense_95/bias/Read/ReadVariableOpReadVariableOpdnn__flatten/dense_95/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
�
-Adam/dnn__flatten/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*>
shared_name/-Adam/dnn__flatten/batch_normalization/gamma/m
�
AAdam/dnn__flatten/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp-Adam/dnn__flatten/batch_normalization/gamma/m*
_output_shapes
:i*
dtype0
�
,Adam/dnn__flatten/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*=
shared_name.,Adam/dnn__flatten/batch_normalization/beta/m
�
@Adam/dnn__flatten/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOp,Adam/dnn__flatten/batch_normalization/beta/m*
_output_shapes
:i*
dtype0
�
#Adam/dnn__flatten/dense_92/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	i�*4
shared_name%#Adam/dnn__flatten/dense_92/kernel/m
�
7Adam/dnn__flatten/dense_92/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/dnn__flatten/dense_92/kernel/m*
_output_shapes
:	i�*
dtype0
�
!Adam/dnn__flatten/dense_92/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adam/dnn__flatten/dense_92/bias/m
�
5Adam/dnn__flatten/dense_92/bias/m/Read/ReadVariableOpReadVariableOp!Adam/dnn__flatten/dense_92/bias/m*
_output_shapes	
:�*
dtype0
�
#Adam/dnn__flatten/dense_93/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*4
shared_name%#Adam/dnn__flatten/dense_93/kernel/m
�
7Adam/dnn__flatten/dense_93/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/dnn__flatten/dense_93/kernel/m* 
_output_shapes
:
��*
dtype0
�
!Adam/dnn__flatten/dense_93/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adam/dnn__flatten/dense_93/bias/m
�
5Adam/dnn__flatten/dense_93/bias/m/Read/ReadVariableOpReadVariableOp!Adam/dnn__flatten/dense_93/bias/m*
_output_shapes	
:�*
dtype0
�
#Adam/dnn__flatten/dense_94/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*4
shared_name%#Adam/dnn__flatten/dense_94/kernel/m
�
7Adam/dnn__flatten/dense_94/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/dnn__flatten/dense_94/kernel/m* 
_output_shapes
:
��*
dtype0
�
!Adam/dnn__flatten/dense_94/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adam/dnn__flatten/dense_94/bias/m
�
5Adam/dnn__flatten/dense_94/bias/m/Read/ReadVariableOpReadVariableOp!Adam/dnn__flatten/dense_94/bias/m*
_output_shapes	
:�*
dtype0
�
#Adam/dnn__flatten/dense_95/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*4
shared_name%#Adam/dnn__flatten/dense_95/kernel/m
�
7Adam/dnn__flatten/dense_95/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/dnn__flatten/dense_95/kernel/m*
_output_shapes
:	�*
dtype0
�
!Adam/dnn__flatten/dense_95/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/dnn__flatten/dense_95/bias/m
�
5Adam/dnn__flatten/dense_95/bias/m/Read/ReadVariableOpReadVariableOp!Adam/dnn__flatten/dense_95/bias/m*
_output_shapes
:*
dtype0
�
-Adam/dnn__flatten/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*>
shared_name/-Adam/dnn__flatten/batch_normalization/gamma/v
�
AAdam/dnn__flatten/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp-Adam/dnn__flatten/batch_normalization/gamma/v*
_output_shapes
:i*
dtype0
�
,Adam/dnn__flatten/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*=
shared_name.,Adam/dnn__flatten/batch_normalization/beta/v
�
@Adam/dnn__flatten/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOp,Adam/dnn__flatten/batch_normalization/beta/v*
_output_shapes
:i*
dtype0
�
#Adam/dnn__flatten/dense_92/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	i�*4
shared_name%#Adam/dnn__flatten/dense_92/kernel/v
�
7Adam/dnn__flatten/dense_92/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/dnn__flatten/dense_92/kernel/v*
_output_shapes
:	i�*
dtype0
�
!Adam/dnn__flatten/dense_92/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adam/dnn__flatten/dense_92/bias/v
�
5Adam/dnn__flatten/dense_92/bias/v/Read/ReadVariableOpReadVariableOp!Adam/dnn__flatten/dense_92/bias/v*
_output_shapes	
:�*
dtype0
�
#Adam/dnn__flatten/dense_93/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*4
shared_name%#Adam/dnn__flatten/dense_93/kernel/v
�
7Adam/dnn__flatten/dense_93/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/dnn__flatten/dense_93/kernel/v* 
_output_shapes
:
��*
dtype0
�
!Adam/dnn__flatten/dense_93/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adam/dnn__flatten/dense_93/bias/v
�
5Adam/dnn__flatten/dense_93/bias/v/Read/ReadVariableOpReadVariableOp!Adam/dnn__flatten/dense_93/bias/v*
_output_shapes	
:�*
dtype0
�
#Adam/dnn__flatten/dense_94/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*4
shared_name%#Adam/dnn__flatten/dense_94/kernel/v
�
7Adam/dnn__flatten/dense_94/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/dnn__flatten/dense_94/kernel/v* 
_output_shapes
:
��*
dtype0
�
!Adam/dnn__flatten/dense_94/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adam/dnn__flatten/dense_94/bias/v
�
5Adam/dnn__flatten/dense_94/bias/v/Read/ReadVariableOpReadVariableOp!Adam/dnn__flatten/dense_94/bias/v*
_output_shapes	
:�*
dtype0
�
#Adam/dnn__flatten/dense_95/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*4
shared_name%#Adam/dnn__flatten/dense_95/kernel/v
�
7Adam/dnn__flatten/dense_95/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/dnn__flatten/dense_95/kernel/v*
_output_shapes
:	�*
dtype0
�
!Adam/dnn__flatten/dense_95/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/dnn__flatten/dense_95/bias/v
�
5Adam/dnn__flatten/dense_95/bias/v/Read/ReadVariableOpReadVariableOp!Adam/dnn__flatten/dense_95/bias/v*
_output_shapes
:*
dtype0
�
0Adam/dnn__flatten/batch_normalization/gamma/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*A
shared_name20Adam/dnn__flatten/batch_normalization/gamma/vhat
�
DAdam/dnn__flatten/batch_normalization/gamma/vhat/Read/ReadVariableOpReadVariableOp0Adam/dnn__flatten/batch_normalization/gamma/vhat*
_output_shapes
:i*
dtype0
�
/Adam/dnn__flatten/batch_normalization/beta/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*@
shared_name1/Adam/dnn__flatten/batch_normalization/beta/vhat
�
CAdam/dnn__flatten/batch_normalization/beta/vhat/Read/ReadVariableOpReadVariableOp/Adam/dnn__flatten/batch_normalization/beta/vhat*
_output_shapes
:i*
dtype0
�
&Adam/dnn__flatten/dense_92/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	i�*7
shared_name(&Adam/dnn__flatten/dense_92/kernel/vhat
�
:Adam/dnn__flatten/dense_92/kernel/vhat/Read/ReadVariableOpReadVariableOp&Adam/dnn__flatten/dense_92/kernel/vhat*
_output_shapes
:	i�*
dtype0
�
$Adam/dnn__flatten/dense_92/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*5
shared_name&$Adam/dnn__flatten/dense_92/bias/vhat
�
8Adam/dnn__flatten/dense_92/bias/vhat/Read/ReadVariableOpReadVariableOp$Adam/dnn__flatten/dense_92/bias/vhat*
_output_shapes	
:�*
dtype0
�
&Adam/dnn__flatten/dense_93/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*7
shared_name(&Adam/dnn__flatten/dense_93/kernel/vhat
�
:Adam/dnn__flatten/dense_93/kernel/vhat/Read/ReadVariableOpReadVariableOp&Adam/dnn__flatten/dense_93/kernel/vhat* 
_output_shapes
:
��*
dtype0
�
$Adam/dnn__flatten/dense_93/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*5
shared_name&$Adam/dnn__flatten/dense_93/bias/vhat
�
8Adam/dnn__flatten/dense_93/bias/vhat/Read/ReadVariableOpReadVariableOp$Adam/dnn__flatten/dense_93/bias/vhat*
_output_shapes	
:�*
dtype0
�
&Adam/dnn__flatten/dense_94/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*7
shared_name(&Adam/dnn__flatten/dense_94/kernel/vhat
�
:Adam/dnn__flatten/dense_94/kernel/vhat/Read/ReadVariableOpReadVariableOp&Adam/dnn__flatten/dense_94/kernel/vhat* 
_output_shapes
:
��*
dtype0
�
$Adam/dnn__flatten/dense_94/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*5
shared_name&$Adam/dnn__flatten/dense_94/bias/vhat
�
8Adam/dnn__flatten/dense_94/bias/vhat/Read/ReadVariableOpReadVariableOp$Adam/dnn__flatten/dense_94/bias/vhat*
_output_shapes	
:�*
dtype0
�
&Adam/dnn__flatten/dense_95/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*7
shared_name(&Adam/dnn__flatten/dense_95/kernel/vhat
�
:Adam/dnn__flatten/dense_95/kernel/vhat/Read/ReadVariableOpReadVariableOp&Adam/dnn__flatten/dense_95/kernel/vhat*
_output_shapes
:	�*
dtype0
�
$Adam/dnn__flatten/dense_95/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/dnn__flatten/dense_95/bias/vhat
�
8Adam/dnn__flatten/dense_95/bias/vhat/Read/ReadVariableOpReadVariableOp$Adam/dnn__flatten/dense_95/bias/vhat*
_output_shapes
:*
dtype0

NoOpNoOp
�A
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�A
value�AB�A B�A
�
	model

batch_norm
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures

	0

1
2
3
�
axis
	gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
�
iter

beta_1

beta_2
	decay
learning_ratemVmWmXmYmZm[m\ m]!m^"m_v`vavbvcvdvevf vg!vh"vi
vhatj
vhatk
vhatl
vhatm
vhatn
vhato
vhatp
 vhatq
!vhatr
"vhats
V
0
1
2
3
4
 5
!6
"7
8
9
10
11
F
0
1
2
3
4
 5
!6
"7
8
9
 
�
	variables
trainable_variables
#layer_regularization_losses
$metrics
%layer_metrics
regularization_losses

&layers
'non_trainable_variables
 
h

kernel
bias
(	variables
)trainable_variables
*regularization_losses
+	keras_api
h

kernel
bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
h

kernel
 bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
h

!kernel
"bias
4	variables
5trainable_variables
6regularization_losses
7	keras_api
 
ge
VARIABLE_VALUE&dnn__flatten/batch_normalization/gamma+batch_norm/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE%dnn__flatten/batch_normalization/beta*batch_norm/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE,dnn__flatten/batch_normalization/moving_mean1batch_norm/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE0dnn__flatten/batch_normalization/moving_variance5batch_norm/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3

0
1
 
�
	variables
trainable_variables
8layer_regularization_losses
9metrics
:layer_metrics
regularization_losses

;layers
<non_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdnn__flatten/dense_92/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdnn__flatten/dense_92/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdnn__flatten/dense_93/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdnn__flatten/dense_93/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdnn__flatten/dense_94/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdnn__flatten/dense_94/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdnn__flatten/dense_95/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdnn__flatten/dense_95/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
 

=0
 
#
	0

1
2
3
4

0
1

0
1

0
1
 
�
(	variables
)trainable_variables
>layer_regularization_losses
?metrics
@layer_metrics
*regularization_losses

Alayers
Bnon_trainable_variables

0
1

0
1
 
�
,	variables
-trainable_variables
Clayer_regularization_losses
Dmetrics
Elayer_metrics
.regularization_losses

Flayers
Gnon_trainable_variables

0
 1

0
 1
 
�
0	variables
1trainable_variables
Hlayer_regularization_losses
Imetrics
Jlayer_metrics
2regularization_losses

Klayers
Lnon_trainable_variables

!0
"1

!0
"1
 
�
4	variables
5trainable_variables
Mlayer_regularization_losses
Nmetrics
Olayer_metrics
6regularization_losses

Players
Qnon_trainable_variables
 
 
 
 

0
1
4
	Rtotal
	Scount
T	variables
U	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

R0
S1

T	variables
��
VARIABLE_VALUE-Adam/dnn__flatten/batch_normalization/gamma/mGbatch_norm/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/dnn__flatten/batch_normalization/beta/mFbatch_norm/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE#Adam/dnn__flatten/dense_92/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/dnn__flatten/dense_92/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE#Adam/dnn__flatten/dense_93/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/dnn__flatten/dense_93/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE#Adam/dnn__flatten/dense_94/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/dnn__flatten/dense_94/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE#Adam/dnn__flatten/dense_95/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/dnn__flatten/dense_95/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE-Adam/dnn__flatten/batch_normalization/gamma/vGbatch_norm/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/dnn__flatten/batch_normalization/beta/vFbatch_norm/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE#Adam/dnn__flatten/dense_92/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/dnn__flatten/dense_92/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE#Adam/dnn__flatten/dense_93/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/dnn__flatten/dense_93/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE#Adam/dnn__flatten/dense_94/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/dnn__flatten/dense_94/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE#Adam/dnn__flatten/dense_95/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/dnn__flatten/dense_95/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE0Adam/dnn__flatten/batch_normalization/gamma/vhatJbatch_norm/gamma/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE/Adam/dnn__flatten/batch_normalization/beta/vhatIbatch_norm/beta/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUE&Adam/dnn__flatten/dense_92/kernel/vhatEvariables/0/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE$Adam/dnn__flatten/dense_92/bias/vhatEvariables/1/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUE&Adam/dnn__flatten/dense_93/kernel/vhatEvariables/2/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE$Adam/dnn__flatten/dense_93/bias/vhatEvariables/3/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUE&Adam/dnn__flatten/dense_94/kernel/vhatEvariables/4/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE$Adam/dnn__flatten/dense_94/bias/vhatEvariables/5/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUE&Adam/dnn__flatten/dense_95/kernel/vhatEvariables/6/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE$Adam/dnn__flatten/dense_95/bias/vhatEvariables/7/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_1Placeholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1,dnn__flatten/batch_normalization/moving_mean0dnn__flatten/batch_normalization/moving_variance%dnn__flatten/batch_normalization/beta&dnn__flatten/batch_normalization/gammadnn__flatten/dense_92/kerneldnn__flatten/dense_92/biasdnn__flatten/dense_93/kerneldnn__flatten/dense_93/biasdnn__flatten/dense_94/kerneldnn__flatten/dense_94/biasdnn__flatten/dense_95/kerneldnn__flatten/dense_95/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8� *.
f)R'
%__inference_signature_wrapper_2322993
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename:dnn__flatten/batch_normalization/gamma/Read/ReadVariableOp9dnn__flatten/batch_normalization/beta/Read/ReadVariableOp@dnn__flatten/batch_normalization/moving_mean/Read/ReadVariableOpDdnn__flatten/batch_normalization/moving_variance/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp0dnn__flatten/dense_92/kernel/Read/ReadVariableOp.dnn__flatten/dense_92/bias/Read/ReadVariableOp0dnn__flatten/dense_93/kernel/Read/ReadVariableOp.dnn__flatten/dense_93/bias/Read/ReadVariableOp0dnn__flatten/dense_94/kernel/Read/ReadVariableOp.dnn__flatten/dense_94/bias/Read/ReadVariableOp0dnn__flatten/dense_95/kernel/Read/ReadVariableOp.dnn__flatten/dense_95/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpAAdam/dnn__flatten/batch_normalization/gamma/m/Read/ReadVariableOp@Adam/dnn__flatten/batch_normalization/beta/m/Read/ReadVariableOp7Adam/dnn__flatten/dense_92/kernel/m/Read/ReadVariableOp5Adam/dnn__flatten/dense_92/bias/m/Read/ReadVariableOp7Adam/dnn__flatten/dense_93/kernel/m/Read/ReadVariableOp5Adam/dnn__flatten/dense_93/bias/m/Read/ReadVariableOp7Adam/dnn__flatten/dense_94/kernel/m/Read/ReadVariableOp5Adam/dnn__flatten/dense_94/bias/m/Read/ReadVariableOp7Adam/dnn__flatten/dense_95/kernel/m/Read/ReadVariableOp5Adam/dnn__flatten/dense_95/bias/m/Read/ReadVariableOpAAdam/dnn__flatten/batch_normalization/gamma/v/Read/ReadVariableOp@Adam/dnn__flatten/batch_normalization/beta/v/Read/ReadVariableOp7Adam/dnn__flatten/dense_92/kernel/v/Read/ReadVariableOp5Adam/dnn__flatten/dense_92/bias/v/Read/ReadVariableOp7Adam/dnn__flatten/dense_93/kernel/v/Read/ReadVariableOp5Adam/dnn__flatten/dense_93/bias/v/Read/ReadVariableOp7Adam/dnn__flatten/dense_94/kernel/v/Read/ReadVariableOp5Adam/dnn__flatten/dense_94/bias/v/Read/ReadVariableOp7Adam/dnn__flatten/dense_95/kernel/v/Read/ReadVariableOp5Adam/dnn__flatten/dense_95/bias/v/Read/ReadVariableOpDAdam/dnn__flatten/batch_normalization/gamma/vhat/Read/ReadVariableOpCAdam/dnn__flatten/batch_normalization/beta/vhat/Read/ReadVariableOp:Adam/dnn__flatten/dense_92/kernel/vhat/Read/ReadVariableOp8Adam/dnn__flatten/dense_92/bias/vhat/Read/ReadVariableOp:Adam/dnn__flatten/dense_93/kernel/vhat/Read/ReadVariableOp8Adam/dnn__flatten/dense_93/bias/vhat/Read/ReadVariableOp:Adam/dnn__flatten/dense_94/kernel/vhat/Read/ReadVariableOp8Adam/dnn__flatten/dense_94/bias/vhat/Read/ReadVariableOp:Adam/dnn__flatten/dense_95/kernel/vhat/Read/ReadVariableOp8Adam/dnn__flatten/dense_95/bias/vhat/Read/ReadVariableOpConst*>
Tin7
523	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *)
f$R"
 __inference__traced_save_2323667
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename&dnn__flatten/batch_normalization/gamma%dnn__flatten/batch_normalization/beta,dnn__flatten/batch_normalization/moving_mean0dnn__flatten/batch_normalization/moving_variance	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratednn__flatten/dense_92/kerneldnn__flatten/dense_92/biasdnn__flatten/dense_93/kerneldnn__flatten/dense_93/biasdnn__flatten/dense_94/kerneldnn__flatten/dense_94/biasdnn__flatten/dense_95/kerneldnn__flatten/dense_95/biastotalcount-Adam/dnn__flatten/batch_normalization/gamma/m,Adam/dnn__flatten/batch_normalization/beta/m#Adam/dnn__flatten/dense_92/kernel/m!Adam/dnn__flatten/dense_92/bias/m#Adam/dnn__flatten/dense_93/kernel/m!Adam/dnn__flatten/dense_93/bias/m#Adam/dnn__flatten/dense_94/kernel/m!Adam/dnn__flatten/dense_94/bias/m#Adam/dnn__flatten/dense_95/kernel/m!Adam/dnn__flatten/dense_95/bias/m-Adam/dnn__flatten/batch_normalization/gamma/v,Adam/dnn__flatten/batch_normalization/beta/v#Adam/dnn__flatten/dense_92/kernel/v!Adam/dnn__flatten/dense_92/bias/v#Adam/dnn__flatten/dense_93/kernel/v!Adam/dnn__flatten/dense_93/bias/v#Adam/dnn__flatten/dense_94/kernel/v!Adam/dnn__flatten/dense_94/bias/v#Adam/dnn__flatten/dense_95/kernel/v!Adam/dnn__flatten/dense_95/bias/v0Adam/dnn__flatten/batch_normalization/gamma/vhat/Adam/dnn__flatten/batch_normalization/beta/vhat&Adam/dnn__flatten/dense_92/kernel/vhat$Adam/dnn__flatten/dense_92/bias/vhat&Adam/dnn__flatten/dense_93/kernel/vhat$Adam/dnn__flatten/dense_93/bias/vhat&Adam/dnn__flatten/dense_94/kernel/vhat$Adam/dnn__flatten/dense_94/bias/vhat&Adam/dnn__flatten/dense_95/kernel/vhat$Adam/dnn__flatten/dense_95/bias/vhat*=
Tin6
422*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *,
f'R%
#__inference__traced_restore_2323824��
�

�
E__inference_dense_93_layer_call_and_return_conditional_losses_2323448

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAdd_
	LeakyRelu	LeakyReluBiasAdd:output:0*(
_output_shapes
:����������2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_dense_94_layer_call_fn_2323477

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_94_layer_call_and_return_conditional_losses_23226732
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
E__inference_dense_94_layer_call_and_return_conditional_losses_2323468

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAdd_
	LeakyRelu	LeakyReluBiasAdd:output:0*(
_output_shapes
:����������2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
E__inference_dense_94_layer_call_and_return_conditional_losses_2322673

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAdd_
	LeakyRelu	LeakyReluBiasAdd:output:0*(
_output_shapes
:����������2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�D
�	
I__inference_dnn__flatten_layer_call_and_return_conditional_losses_2323157
input_1>
0batch_normalization_cast_readvariableop_resource:i@
2batch_normalization_cast_1_readvariableop_resource:i@
2batch_normalization_cast_2_readvariableop_resource:i@
2batch_normalization_cast_3_readvariableop_resource:i:
'dense_92_matmul_readvariableop_resource:	i�7
(dense_92_biasadd_readvariableop_resource:	�;
'dense_93_matmul_readvariableop_resource:
��7
(dense_93_biasadd_readvariableop_resource:	�;
'dense_94_matmul_readvariableop_resource:
��7
(dense_94_biasadd_readvariableop_resource:	�:
'dense_95_matmul_readvariableop_resource:	�6
(dense_95_biasadd_readvariableop_resource:
identity��'batch_normalization/Cast/ReadVariableOp�)batch_normalization/Cast_1/ReadVariableOp�)batch_normalization/Cast_2/ReadVariableOp�)batch_normalization/Cast_3/ReadVariableOp�dense_92/BiasAdd/ReadVariableOp�dense_92/MatMul/ReadVariableOp�dense_93/BiasAdd/ReadVariableOp�dense_93/MatMul/ReadVariableOp�dense_94/BiasAdd/ReadVariableOp�dense_94/MatMul/ReadVariableOp�dense_95/BiasAdd/ReadVariableOp�dense_95/MatMul/ReadVariableOpo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����i   2
flatten/Const�
flatten/ReshapeReshapeinput_1flatten/Const:output:0*
T0*'
_output_shapes
:���������i2
flatten/Reshape�
'batch_normalization/Cast/ReadVariableOpReadVariableOp0batch_normalization_cast_readvariableop_resource*
_output_shapes
:i*
dtype02)
'batch_normalization/Cast/ReadVariableOp�
)batch_normalization/Cast_1/ReadVariableOpReadVariableOp2batch_normalization_cast_1_readvariableop_resource*
_output_shapes
:i*
dtype02+
)batch_normalization/Cast_1/ReadVariableOp�
)batch_normalization/Cast_2/ReadVariableOpReadVariableOp2batch_normalization_cast_2_readvariableop_resource*
_output_shapes
:i*
dtype02+
)batch_normalization/Cast_2/ReadVariableOp�
)batch_normalization/Cast_3/ReadVariableOpReadVariableOp2batch_normalization_cast_3_readvariableop_resource*
_output_shapes
:i*
dtype02+
)batch_normalization/Cast_3/ReadVariableOp�
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2%
#batch_normalization/batchnorm/add/y�
!batch_normalization/batchnorm/addAddV21batch_normalization/Cast_1/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:i2#
!batch_normalization/batchnorm/add�
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:i2%
#batch_normalization/batchnorm/Rsqrt�
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:01batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:i2#
!batch_normalization/batchnorm/mul�
#batch_normalization/batchnorm/mul_1Mulflatten/Reshape:output:0%batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������i2%
#batch_normalization/batchnorm/mul_1�
#batch_normalization/batchnorm/mul_2Mul/batch_normalization/Cast/ReadVariableOp:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:i2%
#batch_normalization/batchnorm/mul_2�
!batch_normalization/batchnorm/subSub1batch_normalization/Cast_2/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:i2#
!batch_normalization/batchnorm/sub�
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������i2%
#batch_normalization/batchnorm/add_1�
dense_92/MatMul/ReadVariableOpReadVariableOp'dense_92_matmul_readvariableop_resource*
_output_shapes
:	i�*
dtype02 
dense_92/MatMul/ReadVariableOp�
dense_92/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0&dense_92/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_92/MatMul�
dense_92/BiasAdd/ReadVariableOpReadVariableOp(dense_92_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_92/BiasAdd/ReadVariableOp�
dense_92/BiasAddBiasAdddense_92/MatMul:product:0'dense_92/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_92/BiasAddz
dense_92/LeakyRelu	LeakyReludense_92/BiasAdd:output:0*(
_output_shapes
:����������2
dense_92/LeakyRelu�
dense_93/MatMul/ReadVariableOpReadVariableOp'dense_93_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_93/MatMul/ReadVariableOp�
dense_93/MatMulMatMul dense_92/LeakyRelu:activations:0&dense_93/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_93/MatMul�
dense_93/BiasAdd/ReadVariableOpReadVariableOp(dense_93_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_93/BiasAdd/ReadVariableOp�
dense_93/BiasAddBiasAdddense_93/MatMul:product:0'dense_93/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_93/BiasAddz
dense_93/LeakyRelu	LeakyReludense_93/BiasAdd:output:0*(
_output_shapes
:����������2
dense_93/LeakyRelu�
dense_94/MatMul/ReadVariableOpReadVariableOp'dense_94_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_94/MatMul/ReadVariableOp�
dense_94/MatMulMatMul dense_93/LeakyRelu:activations:0&dense_94/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_94/MatMul�
dense_94/BiasAdd/ReadVariableOpReadVariableOp(dense_94_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_94/BiasAdd/ReadVariableOp�
dense_94/BiasAddBiasAdddense_94/MatMul:product:0'dense_94/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_94/BiasAddz
dense_94/LeakyRelu	LeakyReludense_94/BiasAdd:output:0*(
_output_shapes
:����������2
dense_94/LeakyRelu�
dense_95/MatMul/ReadVariableOpReadVariableOp'dense_95_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_95/MatMul/ReadVariableOp�
dense_95/MatMulMatMul dense_94/LeakyRelu:activations:0&dense_95/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_95/MatMul�
dense_95/BiasAdd/ReadVariableOpReadVariableOp(dense_95_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_95/BiasAdd/ReadVariableOp�
dense_95/BiasAddBiasAdddense_95/MatMul:product:0'dense_95/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_95/BiasAdd|
dense_95/SoftmaxSoftmaxdense_95/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_95/Softmax�
IdentityIdentitydense_95/Softmax:softmax:0(^batch_normalization/Cast/ReadVariableOp*^batch_normalization/Cast_1/ReadVariableOp*^batch_normalization/Cast_2/ReadVariableOp*^batch_normalization/Cast_3/ReadVariableOp ^dense_92/BiasAdd/ReadVariableOp^dense_92/MatMul/ReadVariableOp ^dense_93/BiasAdd/ReadVariableOp^dense_93/MatMul/ReadVariableOp ^dense_94/BiasAdd/ReadVariableOp^dense_94/MatMul/ReadVariableOp ^dense_95/BiasAdd/ReadVariableOp^dense_95/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 2R
'batch_normalization/Cast/ReadVariableOp'batch_normalization/Cast/ReadVariableOp2V
)batch_normalization/Cast_1/ReadVariableOp)batch_normalization/Cast_1/ReadVariableOp2V
)batch_normalization/Cast_2/ReadVariableOp)batch_normalization/Cast_2/ReadVariableOp2V
)batch_normalization/Cast_3/ReadVariableOp)batch_normalization/Cast_3/ReadVariableOp2B
dense_92/BiasAdd/ReadVariableOpdense_92/BiasAdd/ReadVariableOp2@
dense_92/MatMul/ReadVariableOpdense_92/MatMul/ReadVariableOp2B
dense_93/BiasAdd/ReadVariableOpdense_93/BiasAdd/ReadVariableOp2@
dense_93/MatMul/ReadVariableOpdense_93/MatMul/ReadVariableOp2B
dense_94/BiasAdd/ReadVariableOpdense_94/BiasAdd/ReadVariableOp2@
dense_94/MatMul/ReadVariableOpdense_94/MatMul/ReadVariableOp2B
dense_95/BiasAdd/ReadVariableOpdense_95/BiasAdd/ReadVariableOp2@
dense_95/MatMul/ReadVariableOpdense_95/MatMul/ReadVariableOp:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�k
�
 __inference__traced_save_2323667
file_prefixE
Asavev2_dnn__flatten_batch_normalization_gamma_read_readvariableopD
@savev2_dnn__flatten_batch_normalization_beta_read_readvariableopK
Gsavev2_dnn__flatten_batch_normalization_moving_mean_read_readvariableopO
Ksavev2_dnn__flatten_batch_normalization_moving_variance_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop;
7savev2_dnn__flatten_dense_92_kernel_read_readvariableop9
5savev2_dnn__flatten_dense_92_bias_read_readvariableop;
7savev2_dnn__flatten_dense_93_kernel_read_readvariableop9
5savev2_dnn__flatten_dense_93_bias_read_readvariableop;
7savev2_dnn__flatten_dense_94_kernel_read_readvariableop9
5savev2_dnn__flatten_dense_94_bias_read_readvariableop;
7savev2_dnn__flatten_dense_95_kernel_read_readvariableop9
5savev2_dnn__flatten_dense_95_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopL
Hsavev2_adam_dnn__flatten_batch_normalization_gamma_m_read_readvariableopK
Gsavev2_adam_dnn__flatten_batch_normalization_beta_m_read_readvariableopB
>savev2_adam_dnn__flatten_dense_92_kernel_m_read_readvariableop@
<savev2_adam_dnn__flatten_dense_92_bias_m_read_readvariableopB
>savev2_adam_dnn__flatten_dense_93_kernel_m_read_readvariableop@
<savev2_adam_dnn__flatten_dense_93_bias_m_read_readvariableopB
>savev2_adam_dnn__flatten_dense_94_kernel_m_read_readvariableop@
<savev2_adam_dnn__flatten_dense_94_bias_m_read_readvariableopB
>savev2_adam_dnn__flatten_dense_95_kernel_m_read_readvariableop@
<savev2_adam_dnn__flatten_dense_95_bias_m_read_readvariableopL
Hsavev2_adam_dnn__flatten_batch_normalization_gamma_v_read_readvariableopK
Gsavev2_adam_dnn__flatten_batch_normalization_beta_v_read_readvariableopB
>savev2_adam_dnn__flatten_dense_92_kernel_v_read_readvariableop@
<savev2_adam_dnn__flatten_dense_92_bias_v_read_readvariableopB
>savev2_adam_dnn__flatten_dense_93_kernel_v_read_readvariableop@
<savev2_adam_dnn__flatten_dense_93_bias_v_read_readvariableopB
>savev2_adam_dnn__flatten_dense_94_kernel_v_read_readvariableop@
<savev2_adam_dnn__flatten_dense_94_bias_v_read_readvariableopB
>savev2_adam_dnn__flatten_dense_95_kernel_v_read_readvariableop@
<savev2_adam_dnn__flatten_dense_95_bias_v_read_readvariableopO
Ksavev2_adam_dnn__flatten_batch_normalization_gamma_vhat_read_readvariableopN
Jsavev2_adam_dnn__flatten_batch_normalization_beta_vhat_read_readvariableopE
Asavev2_adam_dnn__flatten_dense_92_kernel_vhat_read_readvariableopC
?savev2_adam_dnn__flatten_dense_92_bias_vhat_read_readvariableopE
Asavev2_adam_dnn__flatten_dense_93_kernel_vhat_read_readvariableopC
?savev2_adam_dnn__flatten_dense_93_bias_vhat_read_readvariableopE
Asavev2_adam_dnn__flatten_dense_94_kernel_vhat_read_readvariableopC
?savev2_adam_dnn__flatten_dense_94_bias_vhat_read_readvariableopE
Asavev2_adam_dnn__flatten_dense_95_kernel_vhat_read_readvariableopC
?savev2_adam_dnn__flatten_dense_95_bias_vhat_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*�
value�B�2B+batch_norm/gamma/.ATTRIBUTES/VARIABLE_VALUEB*batch_norm/beta/.ATTRIBUTES/VARIABLE_VALUEB1batch_norm/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB5batch_norm/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBGbatch_norm/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFbatch_norm/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGbatch_norm/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFbatch_norm/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBJbatch_norm/gamma/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBIbatch_norm/beta/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/0/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/1/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/2/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/3/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/4/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/5/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/6/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/7/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Asavev2_dnn__flatten_batch_normalization_gamma_read_readvariableop@savev2_dnn__flatten_batch_normalization_beta_read_readvariableopGsavev2_dnn__flatten_batch_normalization_moving_mean_read_readvariableopKsavev2_dnn__flatten_batch_normalization_moving_variance_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop7savev2_dnn__flatten_dense_92_kernel_read_readvariableop5savev2_dnn__flatten_dense_92_bias_read_readvariableop7savev2_dnn__flatten_dense_93_kernel_read_readvariableop5savev2_dnn__flatten_dense_93_bias_read_readvariableop7savev2_dnn__flatten_dense_94_kernel_read_readvariableop5savev2_dnn__flatten_dense_94_bias_read_readvariableop7savev2_dnn__flatten_dense_95_kernel_read_readvariableop5savev2_dnn__flatten_dense_95_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopHsavev2_adam_dnn__flatten_batch_normalization_gamma_m_read_readvariableopGsavev2_adam_dnn__flatten_batch_normalization_beta_m_read_readvariableop>savev2_adam_dnn__flatten_dense_92_kernel_m_read_readvariableop<savev2_adam_dnn__flatten_dense_92_bias_m_read_readvariableop>savev2_adam_dnn__flatten_dense_93_kernel_m_read_readvariableop<savev2_adam_dnn__flatten_dense_93_bias_m_read_readvariableop>savev2_adam_dnn__flatten_dense_94_kernel_m_read_readvariableop<savev2_adam_dnn__flatten_dense_94_bias_m_read_readvariableop>savev2_adam_dnn__flatten_dense_95_kernel_m_read_readvariableop<savev2_adam_dnn__flatten_dense_95_bias_m_read_readvariableopHsavev2_adam_dnn__flatten_batch_normalization_gamma_v_read_readvariableopGsavev2_adam_dnn__flatten_batch_normalization_beta_v_read_readvariableop>savev2_adam_dnn__flatten_dense_92_kernel_v_read_readvariableop<savev2_adam_dnn__flatten_dense_92_bias_v_read_readvariableop>savev2_adam_dnn__flatten_dense_93_kernel_v_read_readvariableop<savev2_adam_dnn__flatten_dense_93_bias_v_read_readvariableop>savev2_adam_dnn__flatten_dense_94_kernel_v_read_readvariableop<savev2_adam_dnn__flatten_dense_94_bias_v_read_readvariableop>savev2_adam_dnn__flatten_dense_95_kernel_v_read_readvariableop<savev2_adam_dnn__flatten_dense_95_bias_v_read_readvariableopKsavev2_adam_dnn__flatten_batch_normalization_gamma_vhat_read_readvariableopJsavev2_adam_dnn__flatten_batch_normalization_beta_vhat_read_readvariableopAsavev2_adam_dnn__flatten_dense_92_kernel_vhat_read_readvariableop?savev2_adam_dnn__flatten_dense_92_bias_vhat_read_readvariableopAsavev2_adam_dnn__flatten_dense_93_kernel_vhat_read_readvariableop?savev2_adam_dnn__flatten_dense_93_bias_vhat_read_readvariableopAsavev2_adam_dnn__flatten_dense_94_kernel_vhat_read_readvariableop?savev2_adam_dnn__flatten_dense_94_bias_vhat_read_readvariableopAsavev2_adam_dnn__flatten_dense_95_kernel_vhat_read_readvariableop?savev2_adam_dnn__flatten_dense_95_bias_vhat_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *@
dtypes6
422	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :i:i:i:i: : : : : :	i�:�:
��:�:
��:�:	�:: : :i:i:	i�:�:
��:�:
��:�:	�::i:i:	i�:�:
��:�:
��:�:	�::i:i:	i�:�:
��:�:
��:�:	�:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:i: 

_output_shapes
:i: 

_output_shapes
:i: 

_output_shapes
:i:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :%
!

_output_shapes
:	i�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:i: 

_output_shapes
:i:%!

_output_shapes
:	i�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
:: 

_output_shapes
:i: 

_output_shapes
:i:% !

_output_shapes
:	i�:!!

_output_shapes	
:�:&""
 
_output_shapes
:
��:!#

_output_shapes	
:�:&$"
 
_output_shapes
:
��:!%

_output_shapes	
:�:%&!

_output_shapes
:	�: '

_output_shapes
:: (

_output_shapes
:i: )

_output_shapes
:i:%*!

_output_shapes
:	i�:!+

_output_shapes	
:�:&,"
 
_output_shapes
:
��:!-

_output_shapes	
:�:&."
 
_output_shapes
:
��:!/

_output_shapes	
:�:%0!

_output_shapes
:	�: 1

_output_shapes
::2

_output_shapes
: 
�
�
*__inference_dense_93_layer_call_fn_2323457

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_93_layer_call_and_return_conditional_losses_23226562
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
E__inference_dense_95_layer_call_and_return_conditional_losses_2322690

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�)
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2322532

inputs5
'assignmovingavg_readvariableop_resource:i7
)assignmovingavg_1_readvariableop_resource:i*
cast_readvariableop_resource:i,
cast_1_readvariableop_resource:i
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:i2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������i2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:i*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:i*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:i*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:i2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:i2
AssignMovingAvg/mul�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:i*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:i2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:i2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:i*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:i*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:i2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:i2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:i2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������i2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:i2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:i2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������i2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*'
_output_shapes
:���������i2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������i: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:���������i
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_layer_call_fn_2323417

inputs
unknown:i
	unknown_0:i
	unknown_1:i
	unknown_2:i
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������i*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_23225322
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������i2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������i: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������i
 
_user_specified_nameinputs
�

�
E__inference_dense_92_layer_call_and_return_conditional_losses_2322639

inputs1
matmul_readvariableop_resource:	i�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	i�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAdd_
	LeakyRelu	LeakyReluBiasAdd:output:0*(
_output_shapes
:����������2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������i: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������i
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_layer_call_fn_2323404

inputs
unknown:i
	unknown_0:i
	unknown_1:i
	unknown_2:i
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������i*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_23224722
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������i2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������i: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������i
 
_user_specified_nameinputs
�

�
.__inference_dnn__flatten_layer_call_fn_2323337
input_1
unknown:i
	unknown_0:i
	unknown_1:i
	unknown_2:i
	unknown_3:	i�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:	�

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2*0,1J 8� *R
fMRK
I__inference_dnn__flatten_layer_call_and_return_conditional_losses_23228302
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
*__inference_dense_92_layer_call_fn_2323437

inputs
unknown:	i�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_92_layer_call_and_return_conditional_losses_23226392
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������i: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������i
 
_user_specified_nameinputs
�

�
%__inference_signature_wrapper_2322993
input_1
unknown:i
	unknown_0:i
	unknown_1:i
	unknown_2:i
	unknown_3:	i�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:	�

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8� *+
f&R$
"__inference__wrapped_model_23224482
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�V
�
"__inference__wrapped_model_2322448
input_1K
=dnn__flatten_batch_normalization_cast_readvariableop_resource:iM
?dnn__flatten_batch_normalization_cast_1_readvariableop_resource:iM
?dnn__flatten_batch_normalization_cast_2_readvariableop_resource:iM
?dnn__flatten_batch_normalization_cast_3_readvariableop_resource:iG
4dnn__flatten_dense_92_matmul_readvariableop_resource:	i�D
5dnn__flatten_dense_92_biasadd_readvariableop_resource:	�H
4dnn__flatten_dense_93_matmul_readvariableop_resource:
��D
5dnn__flatten_dense_93_biasadd_readvariableop_resource:	�H
4dnn__flatten_dense_94_matmul_readvariableop_resource:
��D
5dnn__flatten_dense_94_biasadd_readvariableop_resource:	�G
4dnn__flatten_dense_95_matmul_readvariableop_resource:	�C
5dnn__flatten_dense_95_biasadd_readvariableop_resource:
identity��4dnn__flatten/batch_normalization/Cast/ReadVariableOp�6dnn__flatten/batch_normalization/Cast_1/ReadVariableOp�6dnn__flatten/batch_normalization/Cast_2/ReadVariableOp�6dnn__flatten/batch_normalization/Cast_3/ReadVariableOp�,dnn__flatten/dense_92/BiasAdd/ReadVariableOp�+dnn__flatten/dense_92/MatMul/ReadVariableOp�,dnn__flatten/dense_93/BiasAdd/ReadVariableOp�+dnn__flatten/dense_93/MatMul/ReadVariableOp�,dnn__flatten/dense_94/BiasAdd/ReadVariableOp�+dnn__flatten/dense_94/MatMul/ReadVariableOp�,dnn__flatten/dense_95/BiasAdd/ReadVariableOp�+dnn__flatten/dense_95/MatMul/ReadVariableOp�
dnn__flatten/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����i   2
dnn__flatten/flatten/Const�
dnn__flatten/flatten/ReshapeReshapeinput_1#dnn__flatten/flatten/Const:output:0*
T0*'
_output_shapes
:���������i2
dnn__flatten/flatten/Reshape�
4dnn__flatten/batch_normalization/Cast/ReadVariableOpReadVariableOp=dnn__flatten_batch_normalization_cast_readvariableop_resource*
_output_shapes
:i*
dtype026
4dnn__flatten/batch_normalization/Cast/ReadVariableOp�
6dnn__flatten/batch_normalization/Cast_1/ReadVariableOpReadVariableOp?dnn__flatten_batch_normalization_cast_1_readvariableop_resource*
_output_shapes
:i*
dtype028
6dnn__flatten/batch_normalization/Cast_1/ReadVariableOp�
6dnn__flatten/batch_normalization/Cast_2/ReadVariableOpReadVariableOp?dnn__flatten_batch_normalization_cast_2_readvariableop_resource*
_output_shapes
:i*
dtype028
6dnn__flatten/batch_normalization/Cast_2/ReadVariableOp�
6dnn__flatten/batch_normalization/Cast_3/ReadVariableOpReadVariableOp?dnn__flatten_batch_normalization_cast_3_readvariableop_resource*
_output_shapes
:i*
dtype028
6dnn__flatten/batch_normalization/Cast_3/ReadVariableOp�
0dnn__flatten/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:22
0dnn__flatten/batch_normalization/batchnorm/add/y�
.dnn__flatten/batch_normalization/batchnorm/addAddV2>dnn__flatten/batch_normalization/Cast_1/ReadVariableOp:value:09dnn__flatten/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:i20
.dnn__flatten/batch_normalization/batchnorm/add�
0dnn__flatten/batch_normalization/batchnorm/RsqrtRsqrt2dnn__flatten/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:i22
0dnn__flatten/batch_normalization/batchnorm/Rsqrt�
.dnn__flatten/batch_normalization/batchnorm/mulMul4dnn__flatten/batch_normalization/batchnorm/Rsqrt:y:0>dnn__flatten/batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:i20
.dnn__flatten/batch_normalization/batchnorm/mul�
0dnn__flatten/batch_normalization/batchnorm/mul_1Mul%dnn__flatten/flatten/Reshape:output:02dnn__flatten/batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������i22
0dnn__flatten/batch_normalization/batchnorm/mul_1�
0dnn__flatten/batch_normalization/batchnorm/mul_2Mul<dnn__flatten/batch_normalization/Cast/ReadVariableOp:value:02dnn__flatten/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:i22
0dnn__flatten/batch_normalization/batchnorm/mul_2�
.dnn__flatten/batch_normalization/batchnorm/subSub>dnn__flatten/batch_normalization/Cast_2/ReadVariableOp:value:04dnn__flatten/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:i20
.dnn__flatten/batch_normalization/batchnorm/sub�
0dnn__flatten/batch_normalization/batchnorm/add_1AddV24dnn__flatten/batch_normalization/batchnorm/mul_1:z:02dnn__flatten/batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������i22
0dnn__flatten/batch_normalization/batchnorm/add_1�
+dnn__flatten/dense_92/MatMul/ReadVariableOpReadVariableOp4dnn__flatten_dense_92_matmul_readvariableop_resource*
_output_shapes
:	i�*
dtype02-
+dnn__flatten/dense_92/MatMul/ReadVariableOp�
dnn__flatten/dense_92/MatMulMatMul4dnn__flatten/batch_normalization/batchnorm/add_1:z:03dnn__flatten/dense_92/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dnn__flatten/dense_92/MatMul�
,dnn__flatten/dense_92/BiasAdd/ReadVariableOpReadVariableOp5dnn__flatten_dense_92_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,dnn__flatten/dense_92/BiasAdd/ReadVariableOp�
dnn__flatten/dense_92/BiasAddBiasAdd&dnn__flatten/dense_92/MatMul:product:04dnn__flatten/dense_92/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dnn__flatten/dense_92/BiasAdd�
dnn__flatten/dense_92/LeakyRelu	LeakyRelu&dnn__flatten/dense_92/BiasAdd:output:0*(
_output_shapes
:����������2!
dnn__flatten/dense_92/LeakyRelu�
+dnn__flatten/dense_93/MatMul/ReadVariableOpReadVariableOp4dnn__flatten_dense_93_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+dnn__flatten/dense_93/MatMul/ReadVariableOp�
dnn__flatten/dense_93/MatMulMatMul-dnn__flatten/dense_92/LeakyRelu:activations:03dnn__flatten/dense_93/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dnn__flatten/dense_93/MatMul�
,dnn__flatten/dense_93/BiasAdd/ReadVariableOpReadVariableOp5dnn__flatten_dense_93_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,dnn__flatten/dense_93/BiasAdd/ReadVariableOp�
dnn__flatten/dense_93/BiasAddBiasAdd&dnn__flatten/dense_93/MatMul:product:04dnn__flatten/dense_93/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dnn__flatten/dense_93/BiasAdd�
dnn__flatten/dense_93/LeakyRelu	LeakyRelu&dnn__flatten/dense_93/BiasAdd:output:0*(
_output_shapes
:����������2!
dnn__flatten/dense_93/LeakyRelu�
+dnn__flatten/dense_94/MatMul/ReadVariableOpReadVariableOp4dnn__flatten_dense_94_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+dnn__flatten/dense_94/MatMul/ReadVariableOp�
dnn__flatten/dense_94/MatMulMatMul-dnn__flatten/dense_93/LeakyRelu:activations:03dnn__flatten/dense_94/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dnn__flatten/dense_94/MatMul�
,dnn__flatten/dense_94/BiasAdd/ReadVariableOpReadVariableOp5dnn__flatten_dense_94_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,dnn__flatten/dense_94/BiasAdd/ReadVariableOp�
dnn__flatten/dense_94/BiasAddBiasAdd&dnn__flatten/dense_94/MatMul:product:04dnn__flatten/dense_94/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dnn__flatten/dense_94/BiasAdd�
dnn__flatten/dense_94/LeakyRelu	LeakyRelu&dnn__flatten/dense_94/BiasAdd:output:0*(
_output_shapes
:����������2!
dnn__flatten/dense_94/LeakyRelu�
+dnn__flatten/dense_95/MatMul/ReadVariableOpReadVariableOp4dnn__flatten_dense_95_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02-
+dnn__flatten/dense_95/MatMul/ReadVariableOp�
dnn__flatten/dense_95/MatMulMatMul-dnn__flatten/dense_94/LeakyRelu:activations:03dnn__flatten/dense_95/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dnn__flatten/dense_95/MatMul�
,dnn__flatten/dense_95/BiasAdd/ReadVariableOpReadVariableOp5dnn__flatten_dense_95_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,dnn__flatten/dense_95/BiasAdd/ReadVariableOp�
dnn__flatten/dense_95/BiasAddBiasAdd&dnn__flatten/dense_95/MatMul:product:04dnn__flatten/dense_95/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dnn__flatten/dense_95/BiasAdd�
dnn__flatten/dense_95/SoftmaxSoftmax&dnn__flatten/dense_95/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dnn__flatten/dense_95/Softmax�
IdentityIdentity'dnn__flatten/dense_95/Softmax:softmax:05^dnn__flatten/batch_normalization/Cast/ReadVariableOp7^dnn__flatten/batch_normalization/Cast_1/ReadVariableOp7^dnn__flatten/batch_normalization/Cast_2/ReadVariableOp7^dnn__flatten/batch_normalization/Cast_3/ReadVariableOp-^dnn__flatten/dense_92/BiasAdd/ReadVariableOp,^dnn__flatten/dense_92/MatMul/ReadVariableOp-^dnn__flatten/dense_93/BiasAdd/ReadVariableOp,^dnn__flatten/dense_93/MatMul/ReadVariableOp-^dnn__flatten/dense_94/BiasAdd/ReadVariableOp,^dnn__flatten/dense_94/MatMul/ReadVariableOp-^dnn__flatten/dense_95/BiasAdd/ReadVariableOp,^dnn__flatten/dense_95/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 2l
4dnn__flatten/batch_normalization/Cast/ReadVariableOp4dnn__flatten/batch_normalization/Cast/ReadVariableOp2p
6dnn__flatten/batch_normalization/Cast_1/ReadVariableOp6dnn__flatten/batch_normalization/Cast_1/ReadVariableOp2p
6dnn__flatten/batch_normalization/Cast_2/ReadVariableOp6dnn__flatten/batch_normalization/Cast_2/ReadVariableOp2p
6dnn__flatten/batch_normalization/Cast_3/ReadVariableOp6dnn__flatten/batch_normalization/Cast_3/ReadVariableOp2\
,dnn__flatten/dense_92/BiasAdd/ReadVariableOp,dnn__flatten/dense_92/BiasAdd/ReadVariableOp2Z
+dnn__flatten/dense_92/MatMul/ReadVariableOp+dnn__flatten/dense_92/MatMul/ReadVariableOp2\
,dnn__flatten/dense_93/BiasAdd/ReadVariableOp,dnn__flatten/dense_93/BiasAdd/ReadVariableOp2Z
+dnn__flatten/dense_93/MatMul/ReadVariableOp+dnn__flatten/dense_93/MatMul/ReadVariableOp2\
,dnn__flatten/dense_94/BiasAdd/ReadVariableOp,dnn__flatten/dense_94/BiasAdd/ReadVariableOp2Z
+dnn__flatten/dense_94/MatMul/ReadVariableOp+dnn__flatten/dense_94/MatMul/ReadVariableOp2\
,dnn__flatten/dense_95/BiasAdd/ReadVariableOp,dnn__flatten/dense_95/BiasAdd/ReadVariableOp2Z
+dnn__flatten/dense_95/MatMul/ReadVariableOp+dnn__flatten/dense_95/MatMul/ReadVariableOp:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�D
�	
I__inference_dnn__flatten_layer_call_and_return_conditional_losses_2323043

inputs>
0batch_normalization_cast_readvariableop_resource:i@
2batch_normalization_cast_1_readvariableop_resource:i@
2batch_normalization_cast_2_readvariableop_resource:i@
2batch_normalization_cast_3_readvariableop_resource:i:
'dense_92_matmul_readvariableop_resource:	i�7
(dense_92_biasadd_readvariableop_resource:	�;
'dense_93_matmul_readvariableop_resource:
��7
(dense_93_biasadd_readvariableop_resource:	�;
'dense_94_matmul_readvariableop_resource:
��7
(dense_94_biasadd_readvariableop_resource:	�:
'dense_95_matmul_readvariableop_resource:	�6
(dense_95_biasadd_readvariableop_resource:
identity��'batch_normalization/Cast/ReadVariableOp�)batch_normalization/Cast_1/ReadVariableOp�)batch_normalization/Cast_2/ReadVariableOp�)batch_normalization/Cast_3/ReadVariableOp�dense_92/BiasAdd/ReadVariableOp�dense_92/MatMul/ReadVariableOp�dense_93/BiasAdd/ReadVariableOp�dense_93/MatMul/ReadVariableOp�dense_94/BiasAdd/ReadVariableOp�dense_94/MatMul/ReadVariableOp�dense_95/BiasAdd/ReadVariableOp�dense_95/MatMul/ReadVariableOpo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����i   2
flatten/Const
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*'
_output_shapes
:���������i2
flatten/Reshape�
'batch_normalization/Cast/ReadVariableOpReadVariableOp0batch_normalization_cast_readvariableop_resource*
_output_shapes
:i*
dtype02)
'batch_normalization/Cast/ReadVariableOp�
)batch_normalization/Cast_1/ReadVariableOpReadVariableOp2batch_normalization_cast_1_readvariableop_resource*
_output_shapes
:i*
dtype02+
)batch_normalization/Cast_1/ReadVariableOp�
)batch_normalization/Cast_2/ReadVariableOpReadVariableOp2batch_normalization_cast_2_readvariableop_resource*
_output_shapes
:i*
dtype02+
)batch_normalization/Cast_2/ReadVariableOp�
)batch_normalization/Cast_3/ReadVariableOpReadVariableOp2batch_normalization_cast_3_readvariableop_resource*
_output_shapes
:i*
dtype02+
)batch_normalization/Cast_3/ReadVariableOp�
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2%
#batch_normalization/batchnorm/add/y�
!batch_normalization/batchnorm/addAddV21batch_normalization/Cast_1/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:i2#
!batch_normalization/batchnorm/add�
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:i2%
#batch_normalization/batchnorm/Rsqrt�
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:01batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:i2#
!batch_normalization/batchnorm/mul�
#batch_normalization/batchnorm/mul_1Mulflatten/Reshape:output:0%batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������i2%
#batch_normalization/batchnorm/mul_1�
#batch_normalization/batchnorm/mul_2Mul/batch_normalization/Cast/ReadVariableOp:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:i2%
#batch_normalization/batchnorm/mul_2�
!batch_normalization/batchnorm/subSub1batch_normalization/Cast_2/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:i2#
!batch_normalization/batchnorm/sub�
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������i2%
#batch_normalization/batchnorm/add_1�
dense_92/MatMul/ReadVariableOpReadVariableOp'dense_92_matmul_readvariableop_resource*
_output_shapes
:	i�*
dtype02 
dense_92/MatMul/ReadVariableOp�
dense_92/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0&dense_92/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_92/MatMul�
dense_92/BiasAdd/ReadVariableOpReadVariableOp(dense_92_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_92/BiasAdd/ReadVariableOp�
dense_92/BiasAddBiasAdddense_92/MatMul:product:0'dense_92/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_92/BiasAddz
dense_92/LeakyRelu	LeakyReludense_92/BiasAdd:output:0*(
_output_shapes
:����������2
dense_92/LeakyRelu�
dense_93/MatMul/ReadVariableOpReadVariableOp'dense_93_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_93/MatMul/ReadVariableOp�
dense_93/MatMulMatMul dense_92/LeakyRelu:activations:0&dense_93/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_93/MatMul�
dense_93/BiasAdd/ReadVariableOpReadVariableOp(dense_93_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_93/BiasAdd/ReadVariableOp�
dense_93/BiasAddBiasAdddense_93/MatMul:product:0'dense_93/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_93/BiasAddz
dense_93/LeakyRelu	LeakyReludense_93/BiasAdd:output:0*(
_output_shapes
:����������2
dense_93/LeakyRelu�
dense_94/MatMul/ReadVariableOpReadVariableOp'dense_94_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_94/MatMul/ReadVariableOp�
dense_94/MatMulMatMul dense_93/LeakyRelu:activations:0&dense_94/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_94/MatMul�
dense_94/BiasAdd/ReadVariableOpReadVariableOp(dense_94_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_94/BiasAdd/ReadVariableOp�
dense_94/BiasAddBiasAdddense_94/MatMul:product:0'dense_94/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_94/BiasAddz
dense_94/LeakyRelu	LeakyReludense_94/BiasAdd:output:0*(
_output_shapes
:����������2
dense_94/LeakyRelu�
dense_95/MatMul/ReadVariableOpReadVariableOp'dense_95_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_95/MatMul/ReadVariableOp�
dense_95/MatMulMatMul dense_94/LeakyRelu:activations:0&dense_95/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_95/MatMul�
dense_95/BiasAdd/ReadVariableOpReadVariableOp(dense_95_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_95/BiasAdd/ReadVariableOp�
dense_95/BiasAddBiasAdddense_95/MatMul:product:0'dense_95/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_95/BiasAdd|
dense_95/SoftmaxSoftmaxdense_95/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_95/Softmax�
IdentityIdentitydense_95/Softmax:softmax:0(^batch_normalization/Cast/ReadVariableOp*^batch_normalization/Cast_1/ReadVariableOp*^batch_normalization/Cast_2/ReadVariableOp*^batch_normalization/Cast_3/ReadVariableOp ^dense_92/BiasAdd/ReadVariableOp^dense_92/MatMul/ReadVariableOp ^dense_93/BiasAdd/ReadVariableOp^dense_93/MatMul/ReadVariableOp ^dense_94/BiasAdd/ReadVariableOp^dense_94/MatMul/ReadVariableOp ^dense_95/BiasAdd/ReadVariableOp^dense_95/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 2R
'batch_normalization/Cast/ReadVariableOp'batch_normalization/Cast/ReadVariableOp2V
)batch_normalization/Cast_1/ReadVariableOp)batch_normalization/Cast_1/ReadVariableOp2V
)batch_normalization/Cast_2/ReadVariableOp)batch_normalization/Cast_2/ReadVariableOp2V
)batch_normalization/Cast_3/ReadVariableOp)batch_normalization/Cast_3/ReadVariableOp2B
dense_92/BiasAdd/ReadVariableOpdense_92/BiasAdd/ReadVariableOp2@
dense_92/MatMul/ReadVariableOpdense_92/MatMul/ReadVariableOp2B
dense_93/BiasAdd/ReadVariableOpdense_93/BiasAdd/ReadVariableOp2@
dense_93/MatMul/ReadVariableOpdense_93/MatMul/ReadVariableOp2B
dense_94/BiasAdd/ReadVariableOpdense_94/BiasAdd/ReadVariableOp2@
dense_94/MatMul/ReadVariableOpdense_94/MatMul/ReadVariableOp2B
dense_95/BiasAdd/ReadVariableOpdense_95/BiasAdd/ReadVariableOp2@
dense_95/MatMul/ReadVariableOpdense_95/MatMul/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
.__inference_dnn__flatten_layer_call_fn_2323279

inputs
unknown:i
	unknown_0:i
	unknown_1:i
	unknown_2:i
	unknown_3:	i�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:	�

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8� *R
fMRK
I__inference_dnn__flatten_layer_call_and_return_conditional_losses_23226972
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
E__inference_dense_92_layer_call_and_return_conditional_losses_2323428

inputs1
matmul_readvariableop_resource:	i�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	i�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAdd_
	LeakyRelu	LeakyReluBiasAdd:output:0*(
_output_shapes
:����������2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������i: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������i
 
_user_specified_nameinputs
�

�
E__inference_dense_93_layer_call_and_return_conditional_losses_2322656

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAdd_
	LeakyRelu	LeakyReluBiasAdd:output:0*(
_output_shapes
:����������2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
E__inference_dense_95_layer_call_and_return_conditional_losses_2323488

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�c
�

I__inference_dnn__flatten_layer_call_and_return_conditional_losses_2323221
input_1I
;batch_normalization_assignmovingavg_readvariableop_resource:iK
=batch_normalization_assignmovingavg_1_readvariableop_resource:i>
0batch_normalization_cast_readvariableop_resource:i@
2batch_normalization_cast_1_readvariableop_resource:i:
'dense_92_matmul_readvariableop_resource:	i�7
(dense_92_biasadd_readvariableop_resource:	�;
'dense_93_matmul_readvariableop_resource:
��7
(dense_93_biasadd_readvariableop_resource:	�;
'dense_94_matmul_readvariableop_resource:
��7
(dense_94_biasadd_readvariableop_resource:	�:
'dense_95_matmul_readvariableop_resource:	�6
(dense_95_biasadd_readvariableop_resource:
identity��#batch_normalization/AssignMovingAvg�2batch_normalization/AssignMovingAvg/ReadVariableOp�%batch_normalization/AssignMovingAvg_1�4batch_normalization/AssignMovingAvg_1/ReadVariableOp�'batch_normalization/Cast/ReadVariableOp�)batch_normalization/Cast_1/ReadVariableOp�dense_92/BiasAdd/ReadVariableOp�dense_92/MatMul/ReadVariableOp�dense_93/BiasAdd/ReadVariableOp�dense_93/MatMul/ReadVariableOp�dense_94/BiasAdd/ReadVariableOp�dense_94/MatMul/ReadVariableOp�dense_95/BiasAdd/ReadVariableOp�dense_95/MatMul/ReadVariableOpo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����i   2
flatten/Const�
flatten/ReshapeReshapeinput_1flatten/Const:output:0*
T0*'
_output_shapes
:���������i2
flatten/Reshape�
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 24
2batch_normalization/moments/mean/reduction_indices�
 batch_normalization/moments/meanMeanflatten/Reshape:output:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(2"
 batch_normalization/moments/mean�
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*
_output_shapes

:i2*
(batch_normalization/moments/StopGradient�
-batch_normalization/moments/SquaredDifferenceSquaredDifferenceflatten/Reshape:output:01batch_normalization/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������i2/
-batch_normalization/moments/SquaredDifference�
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 28
6batch_normalization/moments/variance/reduction_indices�
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(2&
$batch_normalization/moments/variance�
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:i*
squeeze_dims
 2%
#batch_normalization/moments/Squeeze�
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:i*
squeeze_dims
 2'
%batch_normalization/moments/Squeeze_1�
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2+
)batch_normalization/AssignMovingAvg/decay�
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp;batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes
:i*
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOp�
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
:i2)
'batch_normalization/AssignMovingAvg/sub�
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:i2)
'batch_normalization/AssignMovingAvg/mul�
#batch_normalization/AssignMovingAvgAssignSubVariableOp;batch_normalization_assignmovingavg_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02%
#batch_normalization/AssignMovingAvg�
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization/AssignMovingAvg_1/decay�
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes
:i*
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOp�
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
:i2+
)batch_normalization/AssignMovingAvg_1/sub�
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:i2+
)batch_normalization/AssignMovingAvg_1/mul�
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization/AssignMovingAvg_1�
'batch_normalization/Cast/ReadVariableOpReadVariableOp0batch_normalization_cast_readvariableop_resource*
_output_shapes
:i*
dtype02)
'batch_normalization/Cast/ReadVariableOp�
)batch_normalization/Cast_1/ReadVariableOpReadVariableOp2batch_normalization_cast_1_readvariableop_resource*
_output_shapes
:i*
dtype02+
)batch_normalization/Cast_1/ReadVariableOp�
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2%
#batch_normalization/batchnorm/add/y�
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:i2#
!batch_normalization/batchnorm/add�
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:i2%
#batch_normalization/batchnorm/Rsqrt�
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:01batch_normalization/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:i2#
!batch_normalization/batchnorm/mul�
#batch_normalization/batchnorm/mul_1Mulflatten/Reshape:output:0%batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������i2%
#batch_normalization/batchnorm/mul_1�
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:i2%
#batch_normalization/batchnorm/mul_2�
!batch_normalization/batchnorm/subSub/batch_normalization/Cast/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:i2#
!batch_normalization/batchnorm/sub�
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������i2%
#batch_normalization/batchnorm/add_1�
dense_92/MatMul/ReadVariableOpReadVariableOp'dense_92_matmul_readvariableop_resource*
_output_shapes
:	i�*
dtype02 
dense_92/MatMul/ReadVariableOp�
dense_92/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0&dense_92/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_92/MatMul�
dense_92/BiasAdd/ReadVariableOpReadVariableOp(dense_92_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_92/BiasAdd/ReadVariableOp�
dense_92/BiasAddBiasAdddense_92/MatMul:product:0'dense_92/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_92/BiasAddz
dense_92/LeakyRelu	LeakyReludense_92/BiasAdd:output:0*(
_output_shapes
:����������2
dense_92/LeakyRelu�
dense_93/MatMul/ReadVariableOpReadVariableOp'dense_93_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_93/MatMul/ReadVariableOp�
dense_93/MatMulMatMul dense_92/LeakyRelu:activations:0&dense_93/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_93/MatMul�
dense_93/BiasAdd/ReadVariableOpReadVariableOp(dense_93_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_93/BiasAdd/ReadVariableOp�
dense_93/BiasAddBiasAdddense_93/MatMul:product:0'dense_93/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_93/BiasAddz
dense_93/LeakyRelu	LeakyReludense_93/BiasAdd:output:0*(
_output_shapes
:����������2
dense_93/LeakyRelu�
dense_94/MatMul/ReadVariableOpReadVariableOp'dense_94_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_94/MatMul/ReadVariableOp�
dense_94/MatMulMatMul dense_93/LeakyRelu:activations:0&dense_94/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_94/MatMul�
dense_94/BiasAdd/ReadVariableOpReadVariableOp(dense_94_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_94/BiasAdd/ReadVariableOp�
dense_94/BiasAddBiasAdddense_94/MatMul:product:0'dense_94/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_94/BiasAddz
dense_94/LeakyRelu	LeakyReludense_94/BiasAdd:output:0*(
_output_shapes
:����������2
dense_94/LeakyRelu�
dense_95/MatMul/ReadVariableOpReadVariableOp'dense_95_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_95/MatMul/ReadVariableOp�
dense_95/MatMulMatMul dense_94/LeakyRelu:activations:0&dense_95/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_95/MatMul�
dense_95/BiasAdd/ReadVariableOpReadVariableOp(dense_95_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_95/BiasAdd/ReadVariableOp�
dense_95/BiasAddBiasAdddense_95/MatMul:product:0'dense_95/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_95/BiasAdd|
dense_95/SoftmaxSoftmaxdense_95/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_95/Softmax�
IdentityIdentitydense_95/Softmax:softmax:0$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp(^batch_normalization/Cast/ReadVariableOp*^batch_normalization/Cast_1/ReadVariableOp ^dense_92/BiasAdd/ReadVariableOp^dense_92/MatMul/ReadVariableOp ^dense_93/BiasAdd/ReadVariableOp^dense_93/MatMul/ReadVariableOp ^dense_94/BiasAdd/ReadVariableOp^dense_94/MatMul/ReadVariableOp ^dense_95/BiasAdd/ReadVariableOp^dense_95/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 2J
#batch_normalization/AssignMovingAvg#batch_normalization/AssignMovingAvg2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2N
%batch_normalization/AssignMovingAvg_1%batch_normalization/AssignMovingAvg_12l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2R
'batch_normalization/Cast/ReadVariableOp'batch_normalization/Cast/ReadVariableOp2V
)batch_normalization/Cast_1/ReadVariableOp)batch_normalization/Cast_1/ReadVariableOp2B
dense_92/BiasAdd/ReadVariableOpdense_92/BiasAdd/ReadVariableOp2@
dense_92/MatMul/ReadVariableOpdense_92/MatMul/ReadVariableOp2B
dense_93/BiasAdd/ReadVariableOpdense_93/BiasAdd/ReadVariableOp2@
dense_93/MatMul/ReadVariableOpdense_93/MatMul/ReadVariableOp2B
dense_94/BiasAdd/ReadVariableOpdense_94/BiasAdd/ReadVariableOp2@
dense_94/MatMul/ReadVariableOpdense_94/MatMul/ReadVariableOp2B
dense_95/BiasAdd/ReadVariableOpdense_95/BiasAdd/ReadVariableOp2@
dense_95/MatMul/ReadVariableOpdense_95/MatMul/ReadVariableOp:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2323357

inputs*
cast_readvariableop_resource:i,
cast_1_readvariableop_resource:i,
cast_2_readvariableop_resource:i,
cast_3_readvariableop_resource:i
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:i*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:i*
dtype02
Cast_1/ReadVariableOp�
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:i*
dtype02
Cast_2/ReadVariableOp�
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:i*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:i2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:i2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:i2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������i2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:i2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:i2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������i2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*'
_output_shapes
:���������i2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������i: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:���������i
 
_user_specified_nameinputs
�

�
.__inference_dnn__flatten_layer_call_fn_2323250
input_1
unknown:i
	unknown_0:i
	unknown_1:i
	unknown_2:i
	unknown_3:	i�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:	�

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8� *R
fMRK
I__inference_dnn__flatten_layer_call_and_return_conditional_losses_23226972
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
*__inference_dense_95_layer_call_fn_2323497

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_95_layer_call_and_return_conditional_losses_23226902
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�!
�
I__inference_dnn__flatten_layer_call_and_return_conditional_losses_2322830

inputs)
batch_normalization_2322800:i)
batch_normalization_2322802:i)
batch_normalization_2322804:i)
batch_normalization_2322806:i#
dense_92_2322809:	i�
dense_92_2322811:	�$
dense_93_2322814:
��
dense_93_2322816:	�$
dense_94_2322819:
��
dense_94_2322821:	�#
dense_95_2322824:	�
dense_95_2322826:
identity��+batch_normalization/StatefulPartitionedCall� dense_92/StatefulPartitionedCall� dense_93/StatefulPartitionedCall� dense_94/StatefulPartitionedCall� dense_95/StatefulPartitionedCallo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����i   2
flatten/Const
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*'
_output_shapes
:���������i2
flatten/Reshape�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallflatten/Reshape:output:0batch_normalization_2322800batch_normalization_2322802batch_normalization_2322804batch_normalization_2322806*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������i*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_23225322-
+batch_normalization/StatefulPartitionedCall�
 dense_92/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_92_2322809dense_92_2322811*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_92_layer_call_and_return_conditional_losses_23226392"
 dense_92/StatefulPartitionedCall�
 dense_93/StatefulPartitionedCallStatefulPartitionedCall)dense_92/StatefulPartitionedCall:output:0dense_93_2322814dense_93_2322816*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_93_layer_call_and_return_conditional_losses_23226562"
 dense_93/StatefulPartitionedCall�
 dense_94/StatefulPartitionedCallStatefulPartitionedCall)dense_93/StatefulPartitionedCall:output:0dense_94_2322819dense_94_2322821*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_94_layer_call_and_return_conditional_losses_23226732"
 dense_94/StatefulPartitionedCall�
 dense_95/StatefulPartitionedCallStatefulPartitionedCall)dense_94/StatefulPartitionedCall:output:0dense_95_2322824dense_95_2322826*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_95_layer_call_and_return_conditional_losses_23226902"
 dense_95/StatefulPartitionedCall�
IdentityIdentity)dense_95/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall!^dense_92/StatefulPartitionedCall!^dense_93/StatefulPartitionedCall!^dense_94/StatefulPartitionedCall!^dense_95/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2D
 dense_92/StatefulPartitionedCall dense_92/StatefulPartitionedCall2D
 dense_93/StatefulPartitionedCall dense_93/StatefulPartitionedCall2D
 dense_94/StatefulPartitionedCall dense_94/StatefulPartitionedCall2D
 dense_95/StatefulPartitionedCall dense_95/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�!
�
I__inference_dnn__flatten_layer_call_and_return_conditional_losses_2322697

inputs)
batch_normalization_2322619:i)
batch_normalization_2322621:i)
batch_normalization_2322623:i)
batch_normalization_2322625:i#
dense_92_2322640:	i�
dense_92_2322642:	�$
dense_93_2322657:
��
dense_93_2322659:	�$
dense_94_2322674:
��
dense_94_2322676:	�#
dense_95_2322691:	�
dense_95_2322693:
identity��+batch_normalization/StatefulPartitionedCall� dense_92/StatefulPartitionedCall� dense_93/StatefulPartitionedCall� dense_94/StatefulPartitionedCall� dense_95/StatefulPartitionedCallo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����i   2
flatten/Const
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*'
_output_shapes
:���������i2
flatten/Reshape�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallflatten/Reshape:output:0batch_normalization_2322619batch_normalization_2322621batch_normalization_2322623batch_normalization_2322625*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������i*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_23224722-
+batch_normalization/StatefulPartitionedCall�
 dense_92/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_92_2322640dense_92_2322642*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_92_layer_call_and_return_conditional_losses_23226392"
 dense_92/StatefulPartitionedCall�
 dense_93/StatefulPartitionedCallStatefulPartitionedCall)dense_92/StatefulPartitionedCall:output:0dense_93_2322657dense_93_2322659*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_93_layer_call_and_return_conditional_losses_23226562"
 dense_93/StatefulPartitionedCall�
 dense_94/StatefulPartitionedCallStatefulPartitionedCall)dense_93/StatefulPartitionedCall:output:0dense_94_2322674dense_94_2322676*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_94_layer_call_and_return_conditional_losses_23226732"
 dense_94/StatefulPartitionedCall�
 dense_95/StatefulPartitionedCallStatefulPartitionedCall)dense_94/StatefulPartitionedCall:output:0dense_95_2322691dense_95_2322693*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_95_layer_call_and_return_conditional_losses_23226902"
 dense_95/StatefulPartitionedCall�
IdentityIdentity)dense_95/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall!^dense_92/StatefulPartitionedCall!^dense_93/StatefulPartitionedCall!^dense_94/StatefulPartitionedCall!^dense_95/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2D
 dense_92/StatefulPartitionedCall dense_92/StatefulPartitionedCall2D
 dense_93/StatefulPartitionedCall dense_93/StatefulPartitionedCall2D
 dense_94/StatefulPartitionedCall dense_94/StatefulPartitionedCall2D
 dense_95/StatefulPartitionedCall dense_95/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�)
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2323391

inputs5
'assignmovingavg_readvariableop_resource:i7
)assignmovingavg_1_readvariableop_resource:i*
cast_readvariableop_resource:i,
cast_1_readvariableop_resource:i
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:i2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������i2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:i*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:i*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:i*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:i2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:i2
AssignMovingAvg/mul�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:i*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:i2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:i2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:i*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:i*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:i2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:i2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:i2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������i2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:i2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:i2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������i2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*'
_output_shapes
:���������i2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������i: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:���������i
 
_user_specified_nameinputs
��
�#
#__inference__traced_restore_2323824
file_prefixE
7assignvariableop_dnn__flatten_batch_normalization_gamma:iF
8assignvariableop_1_dnn__flatten_batch_normalization_beta:iM
?assignvariableop_2_dnn__flatten_batch_normalization_moving_mean:iQ
Cassignvariableop_3_dnn__flatten_batch_normalization_moving_variance:i&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: B
/assignvariableop_9_dnn__flatten_dense_92_kernel:	i�=
.assignvariableop_10_dnn__flatten_dense_92_bias:	�D
0assignvariableop_11_dnn__flatten_dense_93_kernel:
��=
.assignvariableop_12_dnn__flatten_dense_93_bias:	�D
0assignvariableop_13_dnn__flatten_dense_94_kernel:
��=
.assignvariableop_14_dnn__flatten_dense_94_bias:	�C
0assignvariableop_15_dnn__flatten_dense_95_kernel:	�<
.assignvariableop_16_dnn__flatten_dense_95_bias:#
assignvariableop_17_total: #
assignvariableop_18_count: O
Aassignvariableop_19_adam_dnn__flatten_batch_normalization_gamma_m:iN
@assignvariableop_20_adam_dnn__flatten_batch_normalization_beta_m:iJ
7assignvariableop_21_adam_dnn__flatten_dense_92_kernel_m:	i�D
5assignvariableop_22_adam_dnn__flatten_dense_92_bias_m:	�K
7assignvariableop_23_adam_dnn__flatten_dense_93_kernel_m:
��D
5assignvariableop_24_adam_dnn__flatten_dense_93_bias_m:	�K
7assignvariableop_25_adam_dnn__flatten_dense_94_kernel_m:
��D
5assignvariableop_26_adam_dnn__flatten_dense_94_bias_m:	�J
7assignvariableop_27_adam_dnn__flatten_dense_95_kernel_m:	�C
5assignvariableop_28_adam_dnn__flatten_dense_95_bias_m:O
Aassignvariableop_29_adam_dnn__flatten_batch_normalization_gamma_v:iN
@assignvariableop_30_adam_dnn__flatten_batch_normalization_beta_v:iJ
7assignvariableop_31_adam_dnn__flatten_dense_92_kernel_v:	i�D
5assignvariableop_32_adam_dnn__flatten_dense_92_bias_v:	�K
7assignvariableop_33_adam_dnn__flatten_dense_93_kernel_v:
��D
5assignvariableop_34_adam_dnn__flatten_dense_93_bias_v:	�K
7assignvariableop_35_adam_dnn__flatten_dense_94_kernel_v:
��D
5assignvariableop_36_adam_dnn__flatten_dense_94_bias_v:	�J
7assignvariableop_37_adam_dnn__flatten_dense_95_kernel_v:	�C
5assignvariableop_38_adam_dnn__flatten_dense_95_bias_v:R
Dassignvariableop_39_adam_dnn__flatten_batch_normalization_gamma_vhat:iQ
Cassignvariableop_40_adam_dnn__flatten_batch_normalization_beta_vhat:iM
:assignvariableop_41_adam_dnn__flatten_dense_92_kernel_vhat:	i�G
8assignvariableop_42_adam_dnn__flatten_dense_92_bias_vhat:	�N
:assignvariableop_43_adam_dnn__flatten_dense_93_kernel_vhat:
��G
8assignvariableop_44_adam_dnn__flatten_dense_93_bias_vhat:	�N
:assignvariableop_45_adam_dnn__flatten_dense_94_kernel_vhat:
��G
8assignvariableop_46_adam_dnn__flatten_dense_94_bias_vhat:	�M
:assignvariableop_47_adam_dnn__flatten_dense_95_kernel_vhat:	�F
8assignvariableop_48_adam_dnn__flatten_dense_95_bias_vhat:
identity_50��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*�
value�B�2B+batch_norm/gamma/.ATTRIBUTES/VARIABLE_VALUEB*batch_norm/beta/.ATTRIBUTES/VARIABLE_VALUEB1batch_norm/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB5batch_norm/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBGbatch_norm/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFbatch_norm/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGbatch_norm/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFbatch_norm/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBJbatch_norm/gamma/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBIbatch_norm/beta/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/0/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/1/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/2/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/3/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/4/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/5/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/6/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/7/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::*@
dtypes6
422	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp7assignvariableop_dnn__flatten_batch_normalization_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp8assignvariableop_1_dnn__flatten_batch_normalization_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp?assignvariableop_2_dnn__flatten_batch_normalization_moving_meanIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpCassignvariableop_3_dnn__flatten_batch_normalization_moving_varianceIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp/assignvariableop_9_dnn__flatten_dense_92_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp.assignvariableop_10_dnn__flatten_dense_92_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp0assignvariableop_11_dnn__flatten_dense_93_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp.assignvariableop_12_dnn__flatten_dense_93_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp0assignvariableop_13_dnn__flatten_dense_94_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp.assignvariableop_14_dnn__flatten_dense_94_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp0assignvariableop_15_dnn__flatten_dense_95_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp.assignvariableop_16_dnn__flatten_dense_95_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpAassignvariableop_19_adam_dnn__flatten_batch_normalization_gamma_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp@assignvariableop_20_adam_dnn__flatten_batch_normalization_beta_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp7assignvariableop_21_adam_dnn__flatten_dense_92_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp5assignvariableop_22_adam_dnn__flatten_dense_92_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp7assignvariableop_23_adam_dnn__flatten_dense_93_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp5assignvariableop_24_adam_dnn__flatten_dense_93_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp7assignvariableop_25_adam_dnn__flatten_dense_94_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp5assignvariableop_26_adam_dnn__flatten_dense_94_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp7assignvariableop_27_adam_dnn__flatten_dense_95_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp5assignvariableop_28_adam_dnn__flatten_dense_95_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOpAassignvariableop_29_adam_dnn__flatten_batch_normalization_gamma_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp@assignvariableop_30_adam_dnn__flatten_batch_normalization_beta_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp7assignvariableop_31_adam_dnn__flatten_dense_92_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp5assignvariableop_32_adam_dnn__flatten_dense_92_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp7assignvariableop_33_adam_dnn__flatten_dense_93_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp5assignvariableop_34_adam_dnn__flatten_dense_93_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp7assignvariableop_35_adam_dnn__flatten_dense_94_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp5assignvariableop_36_adam_dnn__flatten_dense_94_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp7assignvariableop_37_adam_dnn__flatten_dense_95_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp5assignvariableop_38_adam_dnn__flatten_dense_95_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOpDassignvariableop_39_adam_dnn__flatten_batch_normalization_gamma_vhatIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOpCassignvariableop_40_adam_dnn__flatten_batch_normalization_beta_vhatIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp:assignvariableop_41_adam_dnn__flatten_dense_92_kernel_vhatIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp8assignvariableop_42_adam_dnn__flatten_dense_92_bias_vhatIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp:assignvariableop_43_adam_dnn__flatten_dense_93_kernel_vhatIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp8assignvariableop_44_adam_dnn__flatten_dense_93_bias_vhatIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp:assignvariableop_45_adam_dnn__flatten_dense_94_kernel_vhatIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOp8assignvariableop_46_adam_dnn__flatten_dense_94_bias_vhatIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOp:assignvariableop_47_adam_dnn__flatten_dense_95_kernel_vhatIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOp8assignvariableop_48_adam_dnn__flatten_dense_95_bias_vhatIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_489
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�	
Identity_49Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_49�	
Identity_50IdentityIdentity_49:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_50"#
identity_50Identity_50:output:0*w
_input_shapesf
d: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�c
�

I__inference_dnn__flatten_layer_call_and_return_conditional_losses_2323107

inputsI
;batch_normalization_assignmovingavg_readvariableop_resource:iK
=batch_normalization_assignmovingavg_1_readvariableop_resource:i>
0batch_normalization_cast_readvariableop_resource:i@
2batch_normalization_cast_1_readvariableop_resource:i:
'dense_92_matmul_readvariableop_resource:	i�7
(dense_92_biasadd_readvariableop_resource:	�;
'dense_93_matmul_readvariableop_resource:
��7
(dense_93_biasadd_readvariableop_resource:	�;
'dense_94_matmul_readvariableop_resource:
��7
(dense_94_biasadd_readvariableop_resource:	�:
'dense_95_matmul_readvariableop_resource:	�6
(dense_95_biasadd_readvariableop_resource:
identity��#batch_normalization/AssignMovingAvg�2batch_normalization/AssignMovingAvg/ReadVariableOp�%batch_normalization/AssignMovingAvg_1�4batch_normalization/AssignMovingAvg_1/ReadVariableOp�'batch_normalization/Cast/ReadVariableOp�)batch_normalization/Cast_1/ReadVariableOp�dense_92/BiasAdd/ReadVariableOp�dense_92/MatMul/ReadVariableOp�dense_93/BiasAdd/ReadVariableOp�dense_93/MatMul/ReadVariableOp�dense_94/BiasAdd/ReadVariableOp�dense_94/MatMul/ReadVariableOp�dense_95/BiasAdd/ReadVariableOp�dense_95/MatMul/ReadVariableOpo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����i   2
flatten/Const
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*'
_output_shapes
:���������i2
flatten/Reshape�
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 24
2batch_normalization/moments/mean/reduction_indices�
 batch_normalization/moments/meanMeanflatten/Reshape:output:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(2"
 batch_normalization/moments/mean�
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*
_output_shapes

:i2*
(batch_normalization/moments/StopGradient�
-batch_normalization/moments/SquaredDifferenceSquaredDifferenceflatten/Reshape:output:01batch_normalization/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������i2/
-batch_normalization/moments/SquaredDifference�
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 28
6batch_normalization/moments/variance/reduction_indices�
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(2&
$batch_normalization/moments/variance�
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:i*
squeeze_dims
 2%
#batch_normalization/moments/Squeeze�
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:i*
squeeze_dims
 2'
%batch_normalization/moments/Squeeze_1�
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2+
)batch_normalization/AssignMovingAvg/decay�
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp;batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes
:i*
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOp�
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
:i2)
'batch_normalization/AssignMovingAvg/sub�
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:i2)
'batch_normalization/AssignMovingAvg/mul�
#batch_normalization/AssignMovingAvgAssignSubVariableOp;batch_normalization_assignmovingavg_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02%
#batch_normalization/AssignMovingAvg�
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization/AssignMovingAvg_1/decay�
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes
:i*
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOp�
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
:i2+
)batch_normalization/AssignMovingAvg_1/sub�
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:i2+
)batch_normalization/AssignMovingAvg_1/mul�
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization/AssignMovingAvg_1�
'batch_normalization/Cast/ReadVariableOpReadVariableOp0batch_normalization_cast_readvariableop_resource*
_output_shapes
:i*
dtype02)
'batch_normalization/Cast/ReadVariableOp�
)batch_normalization/Cast_1/ReadVariableOpReadVariableOp2batch_normalization_cast_1_readvariableop_resource*
_output_shapes
:i*
dtype02+
)batch_normalization/Cast_1/ReadVariableOp�
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2%
#batch_normalization/batchnorm/add/y�
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:i2#
!batch_normalization/batchnorm/add�
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:i2%
#batch_normalization/batchnorm/Rsqrt�
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:01batch_normalization/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:i2#
!batch_normalization/batchnorm/mul�
#batch_normalization/batchnorm/mul_1Mulflatten/Reshape:output:0%batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������i2%
#batch_normalization/batchnorm/mul_1�
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:i2%
#batch_normalization/batchnorm/mul_2�
!batch_normalization/batchnorm/subSub/batch_normalization/Cast/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:i2#
!batch_normalization/batchnorm/sub�
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������i2%
#batch_normalization/batchnorm/add_1�
dense_92/MatMul/ReadVariableOpReadVariableOp'dense_92_matmul_readvariableop_resource*
_output_shapes
:	i�*
dtype02 
dense_92/MatMul/ReadVariableOp�
dense_92/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0&dense_92/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_92/MatMul�
dense_92/BiasAdd/ReadVariableOpReadVariableOp(dense_92_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_92/BiasAdd/ReadVariableOp�
dense_92/BiasAddBiasAdddense_92/MatMul:product:0'dense_92/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_92/BiasAddz
dense_92/LeakyRelu	LeakyReludense_92/BiasAdd:output:0*(
_output_shapes
:����������2
dense_92/LeakyRelu�
dense_93/MatMul/ReadVariableOpReadVariableOp'dense_93_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_93/MatMul/ReadVariableOp�
dense_93/MatMulMatMul dense_92/LeakyRelu:activations:0&dense_93/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_93/MatMul�
dense_93/BiasAdd/ReadVariableOpReadVariableOp(dense_93_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_93/BiasAdd/ReadVariableOp�
dense_93/BiasAddBiasAdddense_93/MatMul:product:0'dense_93/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_93/BiasAddz
dense_93/LeakyRelu	LeakyReludense_93/BiasAdd:output:0*(
_output_shapes
:����������2
dense_93/LeakyRelu�
dense_94/MatMul/ReadVariableOpReadVariableOp'dense_94_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_94/MatMul/ReadVariableOp�
dense_94/MatMulMatMul dense_93/LeakyRelu:activations:0&dense_94/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_94/MatMul�
dense_94/BiasAdd/ReadVariableOpReadVariableOp(dense_94_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_94/BiasAdd/ReadVariableOp�
dense_94/BiasAddBiasAdddense_94/MatMul:product:0'dense_94/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_94/BiasAddz
dense_94/LeakyRelu	LeakyReludense_94/BiasAdd:output:0*(
_output_shapes
:����������2
dense_94/LeakyRelu�
dense_95/MatMul/ReadVariableOpReadVariableOp'dense_95_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_95/MatMul/ReadVariableOp�
dense_95/MatMulMatMul dense_94/LeakyRelu:activations:0&dense_95/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_95/MatMul�
dense_95/BiasAdd/ReadVariableOpReadVariableOp(dense_95_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_95/BiasAdd/ReadVariableOp�
dense_95/BiasAddBiasAdddense_95/MatMul:product:0'dense_95/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_95/BiasAdd|
dense_95/SoftmaxSoftmaxdense_95/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_95/Softmax�
IdentityIdentitydense_95/Softmax:softmax:0$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp(^batch_normalization/Cast/ReadVariableOp*^batch_normalization/Cast_1/ReadVariableOp ^dense_92/BiasAdd/ReadVariableOp^dense_92/MatMul/ReadVariableOp ^dense_93/BiasAdd/ReadVariableOp^dense_93/MatMul/ReadVariableOp ^dense_94/BiasAdd/ReadVariableOp^dense_94/MatMul/ReadVariableOp ^dense_95/BiasAdd/ReadVariableOp^dense_95/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 2J
#batch_normalization/AssignMovingAvg#batch_normalization/AssignMovingAvg2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2N
%batch_normalization/AssignMovingAvg_1%batch_normalization/AssignMovingAvg_12l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2R
'batch_normalization/Cast/ReadVariableOp'batch_normalization/Cast/ReadVariableOp2V
)batch_normalization/Cast_1/ReadVariableOp)batch_normalization/Cast_1/ReadVariableOp2B
dense_92/BiasAdd/ReadVariableOpdense_92/BiasAdd/ReadVariableOp2@
dense_92/MatMul/ReadVariableOpdense_92/MatMul/ReadVariableOp2B
dense_93/BiasAdd/ReadVariableOpdense_93/BiasAdd/ReadVariableOp2@
dense_93/MatMul/ReadVariableOpdense_93/MatMul/ReadVariableOp2B
dense_94/BiasAdd/ReadVariableOpdense_94/BiasAdd/ReadVariableOp2@
dense_94/MatMul/ReadVariableOpdense_94/MatMul/ReadVariableOp2B
dense_95/BiasAdd/ReadVariableOpdense_95/BiasAdd/ReadVariableOp2@
dense_95/MatMul/ReadVariableOpdense_95/MatMul/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2322472

inputs*
cast_readvariableop_resource:i,
cast_1_readvariableop_resource:i,
cast_2_readvariableop_resource:i,
cast_3_readvariableop_resource:i
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:i*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:i*
dtype02
Cast_1/ReadVariableOp�
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:i*
dtype02
Cast_2/ReadVariableOp�
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:i*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:i2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:i2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:i2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������i2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:i2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:i2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������i2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*'
_output_shapes
:���������i2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������i: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:���������i
 
_user_specified_nameinputs
�

�
.__inference_dnn__flatten_layer_call_fn_2323308

inputs
unknown:i
	unknown_0:i
	unknown_1:i
	unknown_2:i
	unknown_3:	i�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:	�

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2*0,1J 8� *R
fMRK
I__inference_dnn__flatten_layer_call_and_return_conditional_losses_23228302
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
?
input_14
serving_default_input_1:0���������<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:�
�	
	model

batch_norm
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
*t&call_and_return_all_conditional_losses
u_default_save_signature
v__call__"�
_tf_keras_model�{"name": "dnn__flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "DNN_Flatten", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [32, 15, 7]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "DNN_Flatten"}, "training_config": {"loss": {"class_name": "CategoricalCrossentropy", "config": {"reduction": "auto", "name": "categorical_crossentropy", "from_logits": false, "label_smoothing": 0}, "shared_object_id": 0}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": true}}}}
<
	0

1
2
3"
trackable_list_wrapper
�

axis
	gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
*w&call_and_return_all_conditional_losses
x__call__"�
_tf_keras_layer�{"name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 1}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 2}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 4}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 105}}, "shared_object_id": 6}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 105]}}
�
iter

beta_1

beta_2
	decay
learning_ratemVmWmXmYmZm[m\ m]!m^"m_v`vavbvcvdvevf vg!vh"vi
vhatj
vhatk
vhatl
vhatm
vhatn
vhato
vhatp
 vhatq
!vhatr
"vhats"
	optimizer
v
0
1
2
3
4
 5
!6
"7
8
9
10
11"
trackable_list_wrapper
f
0
1
2
3
4
 5
!6
"7
8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
�
	variables
trainable_variables
#layer_regularization_losses
$metrics
%layer_metrics
regularization_losses

&layers
'non_trainable_variables
v__call__
u_default_save_signature
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
,
yserving_default"
signature_map
�

kernel
bias
(	variables
)trainable_variables
*regularization_losses
+	keras_api
*z&call_and_return_all_conditional_losses
{__call__"�
_tf_keras_layer�{"name": "dense_92", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_92", "trainable": true, "dtype": "float32", "units": 256, "activation": "leaky_relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 105}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 105]}}
�

kernel
bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
*|&call_and_return_all_conditional_losses
}__call__"�
_tf_keras_layer�{"name": "dense_93", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_93", "trainable": true, "dtype": "float32", "units": 256, "activation": "leaky_relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 14}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 256]}}
�

kernel
 bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
*~&call_and_return_all_conditional_losses
__call__"�
_tf_keras_layer�{"name": "dense_94", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_94", "trainable": true, "dtype": "float32", "units": 256, "activation": "leaky_relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 18}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 256]}}
�

!kernel
"bias
4	variables
5trainable_variables
6regularization_losses
7	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense_95", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_95", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 22}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 256]}}
 "
trackable_list_wrapper
4:2i2&dnn__flatten/batch_normalization/gamma
3:1i2%dnn__flatten/batch_normalization/beta
<::i (2,dnn__flatten/batch_normalization/moving_mean
@:>i (20dnn__flatten/batch_normalization/moving_variance
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
	variables
trainable_variables
8layer_regularization_losses
9metrics
:layer_metrics
regularization_losses

;layers
<non_trainable_variables
x__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
/:-	i�2dnn__flatten/dense_92/kernel
):'�2dnn__flatten/dense_92/bias
0:.
��2dnn__flatten/dense_93/kernel
):'�2dnn__flatten/dense_93/bias
0:.
��2dnn__flatten/dense_94/kernel
):'�2dnn__flatten/dense_94/bias
/:-	�2dnn__flatten/dense_95/kernel
(:&2dnn__flatten/dense_95/bias
 "
trackable_list_wrapper
'
=0"
trackable_list_wrapper
 "
trackable_dict_wrapper
C
	0

1
2
3
4"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
(	variables
)trainable_variables
>layer_regularization_losses
?metrics
@layer_metrics
*regularization_losses

Alayers
Bnon_trainable_variables
{__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
,	variables
-trainable_variables
Clayer_regularization_losses
Dmetrics
Elayer_metrics
.regularization_losses

Flayers
Gnon_trainable_variables
}__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
0	variables
1trainable_variables
Hlayer_regularization_losses
Imetrics
Jlayer_metrics
2regularization_losses

Klayers
Lnon_trainable_variables
__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
4	variables
5trainable_variables
Mlayer_regularization_losses
Nmetrics
Olayer_metrics
6regularization_losses

Players
Qnon_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
	Rtotal
	Scount
T	variables
U	keras_api"�
_tf_keras_metric�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 23}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
R0
S1"
trackable_list_wrapper
-
T	variables"
_generic_user_object
9:7i2-Adam/dnn__flatten/batch_normalization/gamma/m
8:6i2,Adam/dnn__flatten/batch_normalization/beta/m
4:2	i�2#Adam/dnn__flatten/dense_92/kernel/m
.:,�2!Adam/dnn__flatten/dense_92/bias/m
5:3
��2#Adam/dnn__flatten/dense_93/kernel/m
.:,�2!Adam/dnn__flatten/dense_93/bias/m
5:3
��2#Adam/dnn__flatten/dense_94/kernel/m
.:,�2!Adam/dnn__flatten/dense_94/bias/m
4:2	�2#Adam/dnn__flatten/dense_95/kernel/m
-:+2!Adam/dnn__flatten/dense_95/bias/m
9:7i2-Adam/dnn__flatten/batch_normalization/gamma/v
8:6i2,Adam/dnn__flatten/batch_normalization/beta/v
4:2	i�2#Adam/dnn__flatten/dense_92/kernel/v
.:,�2!Adam/dnn__flatten/dense_92/bias/v
5:3
��2#Adam/dnn__flatten/dense_93/kernel/v
.:,�2!Adam/dnn__flatten/dense_93/bias/v
5:3
��2#Adam/dnn__flatten/dense_94/kernel/v
.:,�2!Adam/dnn__flatten/dense_94/bias/v
4:2	�2#Adam/dnn__flatten/dense_95/kernel/v
-:+2!Adam/dnn__flatten/dense_95/bias/v
<::i20Adam/dnn__flatten/batch_normalization/gamma/vhat
;:9i2/Adam/dnn__flatten/batch_normalization/beta/vhat
7:5	i�2&Adam/dnn__flatten/dense_92/kernel/vhat
1:/�2$Adam/dnn__flatten/dense_92/bias/vhat
8:6
��2&Adam/dnn__flatten/dense_93/kernel/vhat
1:/�2$Adam/dnn__flatten/dense_93/bias/vhat
8:6
��2&Adam/dnn__flatten/dense_94/kernel/vhat
1:/�2$Adam/dnn__flatten/dense_94/bias/vhat
7:5	�2&Adam/dnn__flatten/dense_95/kernel/vhat
0:.2$Adam/dnn__flatten/dense_95/bias/vhat
�2�
I__inference_dnn__flatten_layer_call_and_return_conditional_losses_2323043
I__inference_dnn__flatten_layer_call_and_return_conditional_losses_2323107
I__inference_dnn__flatten_layer_call_and_return_conditional_losses_2323157
I__inference_dnn__flatten_layer_call_and_return_conditional_losses_2323221�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
"__inference__wrapped_model_2322448�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� **�'
%�"
input_1���������
�2�
.__inference_dnn__flatten_layer_call_fn_2323250
.__inference_dnn__flatten_layer_call_fn_2323279
.__inference_dnn__flatten_layer_call_fn_2323308
.__inference_dnn__flatten_layer_call_fn_2323337�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2323357
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2323391�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
5__inference_batch_normalization_layer_call_fn_2323404
5__inference_batch_normalization_layer_call_fn_2323417�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
%__inference_signature_wrapper_2322993input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_92_layer_call_and_return_conditional_losses_2323428�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_92_layer_call_fn_2323437�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_93_layer_call_and_return_conditional_losses_2323448�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_93_layer_call_fn_2323457�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_94_layer_call_and_return_conditional_losses_2323468�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_94_layer_call_fn_2323477�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_95_layer_call_and_return_conditional_losses_2323488�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_95_layer_call_fn_2323497�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
"__inference__wrapped_model_2322448y !"4�1
*�'
%�"
input_1���������
� "3�0
.
output_1"�
output_1����������
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2323357b3�0
)�&
 �
inputs���������i
p 
� "%�"
�
0���������i
� �
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2323391b3�0
)�&
 �
inputs���������i
p
� "%�"
�
0���������i
� �
5__inference_batch_normalization_layer_call_fn_2323404U3�0
)�&
 �
inputs���������i
p 
� "����������i�
5__inference_batch_normalization_layer_call_fn_2323417U3�0
)�&
 �
inputs���������i
p
� "����������i�
E__inference_dense_92_layer_call_and_return_conditional_losses_2323428]/�,
%�"
 �
inputs���������i
� "&�#
�
0����������
� ~
*__inference_dense_92_layer_call_fn_2323437P/�,
%�"
 �
inputs���������i
� "������������
E__inference_dense_93_layer_call_and_return_conditional_losses_2323448^0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� 
*__inference_dense_93_layer_call_fn_2323457Q0�-
&�#
!�
inputs����������
� "������������
E__inference_dense_94_layer_call_and_return_conditional_losses_2323468^ 0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� 
*__inference_dense_94_layer_call_fn_2323477Q 0�-
&�#
!�
inputs����������
� "������������
E__inference_dense_95_layer_call_and_return_conditional_losses_2323488]!"0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� ~
*__inference_dense_95_layer_call_fn_2323497P!"0�-
&�#
!�
inputs����������
� "�����������
I__inference_dnn__flatten_layer_call_and_return_conditional_losses_2323043n !"7�4
-�*
$�!
inputs���������
p 
� "%�"
�
0���������
� �
I__inference_dnn__flatten_layer_call_and_return_conditional_losses_2323107n !"7�4
-�*
$�!
inputs���������
p
� "%�"
�
0���������
� �
I__inference_dnn__flatten_layer_call_and_return_conditional_losses_2323157o !"8�5
.�+
%�"
input_1���������
p 
� "%�"
�
0���������
� �
I__inference_dnn__flatten_layer_call_and_return_conditional_losses_2323221o !"8�5
.�+
%�"
input_1���������
p
� "%�"
�
0���������
� �
.__inference_dnn__flatten_layer_call_fn_2323250b !"8�5
.�+
%�"
input_1���������
p 
� "�����������
.__inference_dnn__flatten_layer_call_fn_2323279a !"7�4
-�*
$�!
inputs���������
p 
� "�����������
.__inference_dnn__flatten_layer_call_fn_2323308a !"7�4
-�*
$�!
inputs���������
p
� "�����������
.__inference_dnn__flatten_layer_call_fn_2323337b !"8�5
.�+
%�"
input_1���������
p
� "�����������
%__inference_signature_wrapper_2322993� !"?�<
� 
5�2
0
input_1%�"
input_1���������"3�0
.
output_1"�
output_1���������