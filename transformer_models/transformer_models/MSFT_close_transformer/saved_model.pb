Έι;
©ϊ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
list(type)(0
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
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Α
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.02v2.9.0-rc2-42-g8a20d54a3c18‘3
Έ
0multi_head_attention_3/attention_output/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20multi_head_attention_3/attention_output/bias/rms
±
Dmulti_head_attention_3/attention_output/bias/rms/Read/ReadVariableOpReadVariableOp0multi_head_attention_3/attention_output/bias/rms*
_output_shapes
:*
dtype0
Δ
2multi_head_attention_3/attention_output/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*C
shared_name42multi_head_attention_3/attention_output/kernel/rms
½
Fmulti_head_attention_3/attention_output/kernel/rms/Read/ReadVariableOpReadVariableOp2multi_head_attention_3/attention_output/kernel/rms*"
_output_shapes
:2*
dtype0
¦
%multi_head_attention_3/value/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*6
shared_name'%multi_head_attention_3/value/bias/rms

9multi_head_attention_3/value/bias/rms/Read/ReadVariableOpReadVariableOp%multi_head_attention_3/value/bias/rms*
_output_shapes

:2*
dtype0
?
'multi_head_attention_3/value/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*8
shared_name)'multi_head_attention_3/value/kernel/rms
§
;multi_head_attention_3/value/kernel/rms/Read/ReadVariableOpReadVariableOp'multi_head_attention_3/value/kernel/rms*"
_output_shapes
:2*
dtype0
’
#multi_head_attention_3/key/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*4
shared_name%#multi_head_attention_3/key/bias/rms

7multi_head_attention_3/key/bias/rms/Read/ReadVariableOpReadVariableOp#multi_head_attention_3/key/bias/rms*
_output_shapes

:2*
dtype0
ͺ
%multi_head_attention_3/key/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*6
shared_name'%multi_head_attention_3/key/kernel/rms
£
9multi_head_attention_3/key/kernel/rms/Read/ReadVariableOpReadVariableOp%multi_head_attention_3/key/kernel/rms*"
_output_shapes
:2*
dtype0
¦
%multi_head_attention_3/query/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*6
shared_name'%multi_head_attention_3/query/bias/rms

9multi_head_attention_3/query/bias/rms/Read/ReadVariableOpReadVariableOp%multi_head_attention_3/query/bias/rms*
_output_shapes

:2*
dtype0
?
'multi_head_attention_3/query/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*8
shared_name)'multi_head_attention_3/query/kernel/rms
§
;multi_head_attention_3/query/kernel/rms/Read/ReadVariableOpReadVariableOp'multi_head_attention_3/query/kernel/rms*"
_output_shapes
:2*
dtype0
Έ
0multi_head_attention_2/attention_output/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20multi_head_attention_2/attention_output/bias/rms
±
Dmulti_head_attention_2/attention_output/bias/rms/Read/ReadVariableOpReadVariableOp0multi_head_attention_2/attention_output/bias/rms*
_output_shapes
:*
dtype0
Δ
2multi_head_attention_2/attention_output/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*C
shared_name42multi_head_attention_2/attention_output/kernel/rms
½
Fmulti_head_attention_2/attention_output/kernel/rms/Read/ReadVariableOpReadVariableOp2multi_head_attention_2/attention_output/kernel/rms*"
_output_shapes
:2*
dtype0
¦
%multi_head_attention_2/value/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*6
shared_name'%multi_head_attention_2/value/bias/rms

9multi_head_attention_2/value/bias/rms/Read/ReadVariableOpReadVariableOp%multi_head_attention_2/value/bias/rms*
_output_shapes

:2*
dtype0
?
'multi_head_attention_2/value/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*8
shared_name)'multi_head_attention_2/value/kernel/rms
§
;multi_head_attention_2/value/kernel/rms/Read/ReadVariableOpReadVariableOp'multi_head_attention_2/value/kernel/rms*"
_output_shapes
:2*
dtype0
’
#multi_head_attention_2/key/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*4
shared_name%#multi_head_attention_2/key/bias/rms

7multi_head_attention_2/key/bias/rms/Read/ReadVariableOpReadVariableOp#multi_head_attention_2/key/bias/rms*
_output_shapes

:2*
dtype0
ͺ
%multi_head_attention_2/key/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*6
shared_name'%multi_head_attention_2/key/kernel/rms
£
9multi_head_attention_2/key/kernel/rms/Read/ReadVariableOpReadVariableOp%multi_head_attention_2/key/kernel/rms*"
_output_shapes
:2*
dtype0
¦
%multi_head_attention_2/query/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*6
shared_name'%multi_head_attention_2/query/bias/rms

9multi_head_attention_2/query/bias/rms/Read/ReadVariableOpReadVariableOp%multi_head_attention_2/query/bias/rms*
_output_shapes

:2*
dtype0
?
'multi_head_attention_2/query/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*8
shared_name)'multi_head_attention_2/query/kernel/rms
§
;multi_head_attention_2/query/kernel/rms/Read/ReadVariableOpReadVariableOp'multi_head_attention_2/query/kernel/rms*"
_output_shapes
:2*
dtype0
Έ
0multi_head_attention_1/attention_output/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20multi_head_attention_1/attention_output/bias/rms
±
Dmulti_head_attention_1/attention_output/bias/rms/Read/ReadVariableOpReadVariableOp0multi_head_attention_1/attention_output/bias/rms*
_output_shapes
:*
dtype0
Δ
2multi_head_attention_1/attention_output/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*C
shared_name42multi_head_attention_1/attention_output/kernel/rms
½
Fmulti_head_attention_1/attention_output/kernel/rms/Read/ReadVariableOpReadVariableOp2multi_head_attention_1/attention_output/kernel/rms*"
_output_shapes
:2*
dtype0
¦
%multi_head_attention_1/value/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*6
shared_name'%multi_head_attention_1/value/bias/rms

9multi_head_attention_1/value/bias/rms/Read/ReadVariableOpReadVariableOp%multi_head_attention_1/value/bias/rms*
_output_shapes

:2*
dtype0
?
'multi_head_attention_1/value/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*8
shared_name)'multi_head_attention_1/value/kernel/rms
§
;multi_head_attention_1/value/kernel/rms/Read/ReadVariableOpReadVariableOp'multi_head_attention_1/value/kernel/rms*"
_output_shapes
:2*
dtype0
’
#multi_head_attention_1/key/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*4
shared_name%#multi_head_attention_1/key/bias/rms

7multi_head_attention_1/key/bias/rms/Read/ReadVariableOpReadVariableOp#multi_head_attention_1/key/bias/rms*
_output_shapes

:2*
dtype0
ͺ
%multi_head_attention_1/key/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*6
shared_name'%multi_head_attention_1/key/kernel/rms
£
9multi_head_attention_1/key/kernel/rms/Read/ReadVariableOpReadVariableOp%multi_head_attention_1/key/kernel/rms*"
_output_shapes
:2*
dtype0
¦
%multi_head_attention_1/query/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*6
shared_name'%multi_head_attention_1/query/bias/rms

9multi_head_attention_1/query/bias/rms/Read/ReadVariableOpReadVariableOp%multi_head_attention_1/query/bias/rms*
_output_shapes

:2*
dtype0
?
'multi_head_attention_1/query/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*8
shared_name)'multi_head_attention_1/query/kernel/rms
§
;multi_head_attention_1/query/kernel/rms/Read/ReadVariableOpReadVariableOp'multi_head_attention_1/query/kernel/rms*"
_output_shapes
:2*
dtype0
΄
.multi_head_attention/attention_output/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.multi_head_attention/attention_output/bias/rms
­
Bmulti_head_attention/attention_output/bias/rms/Read/ReadVariableOpReadVariableOp.multi_head_attention/attention_output/bias/rms*
_output_shapes
:*
dtype0
ΐ
0multi_head_attention/attention_output/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*A
shared_name20multi_head_attention/attention_output/kernel/rms
Ή
Dmulti_head_attention/attention_output/kernel/rms/Read/ReadVariableOpReadVariableOp0multi_head_attention/attention_output/kernel/rms*"
_output_shapes
:2*
dtype0
’
#multi_head_attention/value/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*4
shared_name%#multi_head_attention/value/bias/rms

7multi_head_attention/value/bias/rms/Read/ReadVariableOpReadVariableOp#multi_head_attention/value/bias/rms*
_output_shapes

:2*
dtype0
ͺ
%multi_head_attention/value/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*6
shared_name'%multi_head_attention/value/kernel/rms
£
9multi_head_attention/value/kernel/rms/Read/ReadVariableOpReadVariableOp%multi_head_attention/value/kernel/rms*"
_output_shapes
:2*
dtype0

!multi_head_attention/key/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*2
shared_name#!multi_head_attention/key/bias/rms

5multi_head_attention/key/bias/rms/Read/ReadVariableOpReadVariableOp!multi_head_attention/key/bias/rms*
_output_shapes

:2*
dtype0
¦
#multi_head_attention/key/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*4
shared_name%#multi_head_attention/key/kernel/rms

7multi_head_attention/key/kernel/rms/Read/ReadVariableOpReadVariableOp#multi_head_attention/key/kernel/rms*"
_output_shapes
:2*
dtype0
’
#multi_head_attention/query/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*4
shared_name%#multi_head_attention/query/bias/rms

7multi_head_attention/query/bias/rms/Read/ReadVariableOpReadVariableOp#multi_head_attention/query/bias/rms*
_output_shapes

:2*
dtype0
ͺ
%multi_head_attention/query/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*6
shared_name'%multi_head_attention/query/kernel/rms
£
9multi_head_attention/query/kernel/rms/Read/ReadVariableOpReadVariableOp%multi_head_attention/query/kernel/rms*"
_output_shapes
:2*
dtype0
x
dense_1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namedense_1/bias/rms
q
$dense_1/bias/rms/Read/ReadVariableOpReadVariableOpdense_1/bias/rms*
_output_shapes
:*
dtype0

dense_1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*#
shared_namedense_1/kernel/rms
z
&dense_1/kernel/rms/Read/ReadVariableOpReadVariableOpdense_1/kernel/rms*
_output_shapes
:	*
dtype0
u
dense/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense/bias/rms
n
"dense/bias/rms/Read/ReadVariableOpReadVariableOpdense/bias/rms*
_output_shapes	
:*
dtype0
~
dense/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Θ*!
shared_namedense/kernel/rms
w
$dense/kernel/rms/Read/ReadVariableOpReadVariableOpdense/kernel/rms* 
_output_shapes
:
Θ*
dtype0
z
conv1d_7/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv1d_7/bias/rms
s
%conv1d_7/bias/rms/Read/ReadVariableOpReadVariableOpconv1d_7/bias/rms*
_output_shapes
:*
dtype0

conv1d_7/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameconv1d_7/kernel/rms

'conv1d_7/kernel/rms/Read/ReadVariableOpReadVariableOpconv1d_7/kernel/rms*"
_output_shapes
:*
dtype0
z
conv1d_6/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv1d_6/bias/rms
s
%conv1d_6/bias/rms/Read/ReadVariableOpReadVariableOpconv1d_6/bias/rms*
_output_shapes
:*
dtype0

conv1d_6/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameconv1d_6/kernel/rms

'conv1d_6/kernel/rms/Read/ReadVariableOpReadVariableOpconv1d_6/kernel/rms*"
_output_shapes
:*
dtype0

layer_normalization_7/beta/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name layer_normalization_7/beta/rms

2layer_normalization_7/beta/rms/Read/ReadVariableOpReadVariableOplayer_normalization_7/beta/rms*
_output_shapes
:*
dtype0

layer_normalization_7/gamma/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!layer_normalization_7/gamma/rms

3layer_normalization_7/gamma/rms/Read/ReadVariableOpReadVariableOplayer_normalization_7/gamma/rms*
_output_shapes
:*
dtype0

layer_normalization_6/beta/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name layer_normalization_6/beta/rms

2layer_normalization_6/beta/rms/Read/ReadVariableOpReadVariableOplayer_normalization_6/beta/rms*
_output_shapes
:*
dtype0

layer_normalization_6/gamma/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!layer_normalization_6/gamma/rms

3layer_normalization_6/gamma/rms/Read/ReadVariableOpReadVariableOplayer_normalization_6/gamma/rms*
_output_shapes
:*
dtype0
z
conv1d_5/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv1d_5/bias/rms
s
%conv1d_5/bias/rms/Read/ReadVariableOpReadVariableOpconv1d_5/bias/rms*
_output_shapes
:*
dtype0

conv1d_5/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameconv1d_5/kernel/rms

'conv1d_5/kernel/rms/Read/ReadVariableOpReadVariableOpconv1d_5/kernel/rms*"
_output_shapes
:*
dtype0
z
conv1d_4/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv1d_4/bias/rms
s
%conv1d_4/bias/rms/Read/ReadVariableOpReadVariableOpconv1d_4/bias/rms*
_output_shapes
:*
dtype0

conv1d_4/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameconv1d_4/kernel/rms

'conv1d_4/kernel/rms/Read/ReadVariableOpReadVariableOpconv1d_4/kernel/rms*"
_output_shapes
:*
dtype0

layer_normalization_5/beta/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name layer_normalization_5/beta/rms

2layer_normalization_5/beta/rms/Read/ReadVariableOpReadVariableOplayer_normalization_5/beta/rms*
_output_shapes
:*
dtype0

layer_normalization_5/gamma/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!layer_normalization_5/gamma/rms

3layer_normalization_5/gamma/rms/Read/ReadVariableOpReadVariableOplayer_normalization_5/gamma/rms*
_output_shapes
:*
dtype0

layer_normalization_4/beta/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name layer_normalization_4/beta/rms

2layer_normalization_4/beta/rms/Read/ReadVariableOpReadVariableOplayer_normalization_4/beta/rms*
_output_shapes
:*
dtype0

layer_normalization_4/gamma/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!layer_normalization_4/gamma/rms

3layer_normalization_4/gamma/rms/Read/ReadVariableOpReadVariableOplayer_normalization_4/gamma/rms*
_output_shapes
:*
dtype0
z
conv1d_3/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv1d_3/bias/rms
s
%conv1d_3/bias/rms/Read/ReadVariableOpReadVariableOpconv1d_3/bias/rms*
_output_shapes
:*
dtype0

conv1d_3/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameconv1d_3/kernel/rms

'conv1d_3/kernel/rms/Read/ReadVariableOpReadVariableOpconv1d_3/kernel/rms*"
_output_shapes
:*
dtype0
z
conv1d_2/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv1d_2/bias/rms
s
%conv1d_2/bias/rms/Read/ReadVariableOpReadVariableOpconv1d_2/bias/rms*
_output_shapes
:*
dtype0

conv1d_2/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameconv1d_2/kernel/rms

'conv1d_2/kernel/rms/Read/ReadVariableOpReadVariableOpconv1d_2/kernel/rms*"
_output_shapes
:*
dtype0

layer_normalization_3/beta/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name layer_normalization_3/beta/rms

2layer_normalization_3/beta/rms/Read/ReadVariableOpReadVariableOplayer_normalization_3/beta/rms*
_output_shapes
:*
dtype0

layer_normalization_3/gamma/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!layer_normalization_3/gamma/rms

3layer_normalization_3/gamma/rms/Read/ReadVariableOpReadVariableOplayer_normalization_3/gamma/rms*
_output_shapes
:*
dtype0

layer_normalization_2/beta/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name layer_normalization_2/beta/rms

2layer_normalization_2/beta/rms/Read/ReadVariableOpReadVariableOplayer_normalization_2/beta/rms*
_output_shapes
:*
dtype0

layer_normalization_2/gamma/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!layer_normalization_2/gamma/rms

3layer_normalization_2/gamma/rms/Read/ReadVariableOpReadVariableOplayer_normalization_2/gamma/rms*
_output_shapes
:*
dtype0
z
conv1d_1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv1d_1/bias/rms
s
%conv1d_1/bias/rms/Read/ReadVariableOpReadVariableOpconv1d_1/bias/rms*
_output_shapes
:*
dtype0

conv1d_1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameconv1d_1/kernel/rms

'conv1d_1/kernel/rms/Read/ReadVariableOpReadVariableOpconv1d_1/kernel/rms*"
_output_shapes
:*
dtype0
v
conv1d/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1d/bias/rms
o
#conv1d/bias/rms/Read/ReadVariableOpReadVariableOpconv1d/bias/rms*
_output_shapes
:*
dtype0

conv1d/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv1d/kernel/rms
{
%conv1d/kernel/rms/Read/ReadVariableOpReadVariableOpconv1d/kernel/rms*"
_output_shapes
:*
dtype0

layer_normalization_1/beta/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name layer_normalization_1/beta/rms

2layer_normalization_1/beta/rms/Read/ReadVariableOpReadVariableOplayer_normalization_1/beta/rms*
_output_shapes
:*
dtype0

layer_normalization_1/gamma/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!layer_normalization_1/gamma/rms

3layer_normalization_1/gamma/rms/Read/ReadVariableOpReadVariableOplayer_normalization_1/gamma/rms*
_output_shapes
:*
dtype0

layer_normalization/beta/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namelayer_normalization/beta/rms

0layer_normalization/beta/rms/Read/ReadVariableOpReadVariableOplayer_normalization/beta/rms*
_output_shapes
:*
dtype0

layer_normalization/gamma/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namelayer_normalization/gamma/rms

1layer_normalization/gamma/rms/Read/ReadVariableOpReadVariableOplayer_normalization/gamma/rms*
_output_shapes
:*
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
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
Z
rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namerho
S
rho/Read/ReadVariableOpReadVariableOprho*
_output_shapes
: *
dtype0
d
momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
momentum
]
momentum/Read/ReadVariableOpReadVariableOpmomentum*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
\
iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameiter
U
iter/Read/ReadVariableOpReadVariableOpiter*
_output_shapes
: *
dtype0	
°
,multi_head_attention_3/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,multi_head_attention_3/attention_output/bias
©
@multi_head_attention_3/attention_output/bias/Read/ReadVariableOpReadVariableOp,multi_head_attention_3/attention_output/bias*
_output_shapes
:*
dtype0
Ό
.multi_head_attention_3/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*?
shared_name0.multi_head_attention_3/attention_output/kernel
΅
Bmulti_head_attention_3/attention_output/kernel/Read/ReadVariableOpReadVariableOp.multi_head_attention_3/attention_output/kernel*"
_output_shapes
:2*
dtype0

!multi_head_attention_3/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*2
shared_name#!multi_head_attention_3/value/bias

5multi_head_attention_3/value/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_3/value/bias*
_output_shapes

:2*
dtype0
¦
#multi_head_attention_3/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*4
shared_name%#multi_head_attention_3/value/kernel

7multi_head_attention_3/value/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_3/value/kernel*"
_output_shapes
:2*
dtype0

multi_head_attention_3/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*0
shared_name!multi_head_attention_3/key/bias

3multi_head_attention_3/key/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention_3/key/bias*
_output_shapes

:2*
dtype0
’
!multi_head_attention_3/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*2
shared_name#!multi_head_attention_3/key/kernel

5multi_head_attention_3/key/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention_3/key/kernel*"
_output_shapes
:2*
dtype0

!multi_head_attention_3/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*2
shared_name#!multi_head_attention_3/query/bias

5multi_head_attention_3/query/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_3/query/bias*
_output_shapes

:2*
dtype0
¦
#multi_head_attention_3/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*4
shared_name%#multi_head_attention_3/query/kernel

7multi_head_attention_3/query/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_3/query/kernel*"
_output_shapes
:2*
dtype0
°
,multi_head_attention_2/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,multi_head_attention_2/attention_output/bias
©
@multi_head_attention_2/attention_output/bias/Read/ReadVariableOpReadVariableOp,multi_head_attention_2/attention_output/bias*
_output_shapes
:*
dtype0
Ό
.multi_head_attention_2/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*?
shared_name0.multi_head_attention_2/attention_output/kernel
΅
Bmulti_head_attention_2/attention_output/kernel/Read/ReadVariableOpReadVariableOp.multi_head_attention_2/attention_output/kernel*"
_output_shapes
:2*
dtype0

!multi_head_attention_2/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*2
shared_name#!multi_head_attention_2/value/bias

5multi_head_attention_2/value/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_2/value/bias*
_output_shapes

:2*
dtype0
¦
#multi_head_attention_2/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*4
shared_name%#multi_head_attention_2/value/kernel

7multi_head_attention_2/value/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_2/value/kernel*"
_output_shapes
:2*
dtype0

multi_head_attention_2/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*0
shared_name!multi_head_attention_2/key/bias

3multi_head_attention_2/key/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention_2/key/bias*
_output_shapes

:2*
dtype0
’
!multi_head_attention_2/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*2
shared_name#!multi_head_attention_2/key/kernel

5multi_head_attention_2/key/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention_2/key/kernel*"
_output_shapes
:2*
dtype0

!multi_head_attention_2/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*2
shared_name#!multi_head_attention_2/query/bias

5multi_head_attention_2/query/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_2/query/bias*
_output_shapes

:2*
dtype0
¦
#multi_head_attention_2/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*4
shared_name%#multi_head_attention_2/query/kernel

7multi_head_attention_2/query/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_2/query/kernel*"
_output_shapes
:2*
dtype0
°
,multi_head_attention_1/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,multi_head_attention_1/attention_output/bias
©
@multi_head_attention_1/attention_output/bias/Read/ReadVariableOpReadVariableOp,multi_head_attention_1/attention_output/bias*
_output_shapes
:*
dtype0
Ό
.multi_head_attention_1/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*?
shared_name0.multi_head_attention_1/attention_output/kernel
΅
Bmulti_head_attention_1/attention_output/kernel/Read/ReadVariableOpReadVariableOp.multi_head_attention_1/attention_output/kernel*"
_output_shapes
:2*
dtype0

!multi_head_attention_1/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*2
shared_name#!multi_head_attention_1/value/bias

5multi_head_attention_1/value/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_1/value/bias*
_output_shapes

:2*
dtype0
¦
#multi_head_attention_1/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*4
shared_name%#multi_head_attention_1/value/kernel

7multi_head_attention_1/value/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_1/value/kernel*"
_output_shapes
:2*
dtype0

multi_head_attention_1/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*0
shared_name!multi_head_attention_1/key/bias

3multi_head_attention_1/key/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention_1/key/bias*
_output_shapes

:2*
dtype0
’
!multi_head_attention_1/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*2
shared_name#!multi_head_attention_1/key/kernel

5multi_head_attention_1/key/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention_1/key/kernel*"
_output_shapes
:2*
dtype0

!multi_head_attention_1/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*2
shared_name#!multi_head_attention_1/query/bias

5multi_head_attention_1/query/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_1/query/bias*
_output_shapes

:2*
dtype0
¦
#multi_head_attention_1/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*4
shared_name%#multi_head_attention_1/query/kernel

7multi_head_attention_1/query/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_1/query/kernel*"
_output_shapes
:2*
dtype0
¬
*multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*multi_head_attention/attention_output/bias
₯
>multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOp*multi_head_attention/attention_output/bias*
_output_shapes
:*
dtype0
Έ
,multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*=
shared_name.,multi_head_attention/attention_output/kernel
±
@multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOp,multi_head_attention/attention_output/kernel*"
_output_shapes
:2*
dtype0

multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*0
shared_name!multi_head_attention/value/bias

3multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/value/bias*
_output_shapes

:2*
dtype0
’
!multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*2
shared_name#!multi_head_attention/value/kernel

5multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention/value/kernel*"
_output_shapes
:2*
dtype0

multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*.
shared_namemulti_head_attention/key/bias

1multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/key/bias*
_output_shapes

:2*
dtype0

multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*0
shared_name!multi_head_attention/key/kernel

3multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOpmulti_head_attention/key/kernel*"
_output_shapes
:2*
dtype0

multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*0
shared_name!multi_head_attention/query/bias

3multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/query/bias*
_output_shapes

:2*
dtype0
’
!multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*2
shared_name#!multi_head_attention/query/kernel

5multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention/query/kernel*"
_output_shapes
:2*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Θ*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
Θ*
dtype0
r
conv1d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_7/bias
k
!conv1d_7/bias/Read/ReadVariableOpReadVariableOpconv1d_7/bias*
_output_shapes
:*
dtype0
~
conv1d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1d_7/kernel
w
#conv1d_7/kernel/Read/ReadVariableOpReadVariableOpconv1d_7/kernel*"
_output_shapes
:*
dtype0
r
conv1d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_6/bias
k
!conv1d_6/bias/Read/ReadVariableOpReadVariableOpconv1d_6/bias*
_output_shapes
:*
dtype0
~
conv1d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1d_6/kernel
w
#conv1d_6/kernel/Read/ReadVariableOpReadVariableOpconv1d_6/kernel*"
_output_shapes
:*
dtype0

layer_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelayer_normalization_7/beta

.layer_normalization_7/beta/Read/ReadVariableOpReadVariableOplayer_normalization_7/beta*
_output_shapes
:*
dtype0

layer_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_7/gamma

/layer_normalization_7/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_7/gamma*
_output_shapes
:*
dtype0

layer_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelayer_normalization_6/beta

.layer_normalization_6/beta/Read/ReadVariableOpReadVariableOplayer_normalization_6/beta*
_output_shapes
:*
dtype0

layer_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_6/gamma

/layer_normalization_6/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_6/gamma*
_output_shapes
:*
dtype0
r
conv1d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_5/bias
k
!conv1d_5/bias/Read/ReadVariableOpReadVariableOpconv1d_5/bias*
_output_shapes
:*
dtype0
~
conv1d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1d_5/kernel
w
#conv1d_5/kernel/Read/ReadVariableOpReadVariableOpconv1d_5/kernel*"
_output_shapes
:*
dtype0
r
conv1d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_4/bias
k
!conv1d_4/bias/Read/ReadVariableOpReadVariableOpconv1d_4/bias*
_output_shapes
:*
dtype0
~
conv1d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1d_4/kernel
w
#conv1d_4/kernel/Read/ReadVariableOpReadVariableOpconv1d_4/kernel*"
_output_shapes
:*
dtype0

layer_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelayer_normalization_5/beta

.layer_normalization_5/beta/Read/ReadVariableOpReadVariableOplayer_normalization_5/beta*
_output_shapes
:*
dtype0

layer_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_5/gamma

/layer_normalization_5/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_5/gamma*
_output_shapes
:*
dtype0

layer_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelayer_normalization_4/beta

.layer_normalization_4/beta/Read/ReadVariableOpReadVariableOplayer_normalization_4/beta*
_output_shapes
:*
dtype0

layer_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_4/gamma

/layer_normalization_4/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_4/gamma*
_output_shapes
:*
dtype0
r
conv1d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_3/bias
k
!conv1d_3/bias/Read/ReadVariableOpReadVariableOpconv1d_3/bias*
_output_shapes
:*
dtype0
~
conv1d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1d_3/kernel
w
#conv1d_3/kernel/Read/ReadVariableOpReadVariableOpconv1d_3/kernel*"
_output_shapes
:*
dtype0
r
conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_2/bias
k
!conv1d_2/bias/Read/ReadVariableOpReadVariableOpconv1d_2/bias*
_output_shapes
:*
dtype0
~
conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1d_2/kernel
w
#conv1d_2/kernel/Read/ReadVariableOpReadVariableOpconv1d_2/kernel*"
_output_shapes
:*
dtype0

layer_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelayer_normalization_3/beta

.layer_normalization_3/beta/Read/ReadVariableOpReadVariableOplayer_normalization_3/beta*
_output_shapes
:*
dtype0

layer_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_3/gamma

/layer_normalization_3/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_3/gamma*
_output_shapes
:*
dtype0

layer_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelayer_normalization_2/beta

.layer_normalization_2/beta/Read/ReadVariableOpReadVariableOplayer_normalization_2/beta*
_output_shapes
:*
dtype0

layer_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_2/gamma

/layer_normalization_2/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_2/gamma*
_output_shapes
:*
dtype0
r
conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_1/bias
k
!conv1d_1/bias/Read/ReadVariableOpReadVariableOpconv1d_1/bias*
_output_shapes
:*
dtype0
~
conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1d_1/kernel
w
#conv1d_1/kernel/Read/ReadVariableOpReadVariableOpconv1d_1/kernel*"
_output_shapes
:*
dtype0
n
conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d/bias
g
conv1d/bias/Read/ReadVariableOpReadVariableOpconv1d/bias*
_output_shapes
:*
dtype0
z
conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d/kernel
s
!conv1d/kernel/Read/ReadVariableOpReadVariableOpconv1d/kernel*"
_output_shapes
:*
dtype0

layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelayer_normalization_1/beta

.layer_normalization_1/beta/Read/ReadVariableOpReadVariableOplayer_normalization_1/beta*
_output_shapes
:*
dtype0

layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_1/gamma

/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_1/gamma*
_output_shapes
:*
dtype0

layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namelayer_normalization/beta

,layer_normalization/beta/Read/ReadVariableOpReadVariableOplayer_normalization/beta*
_output_shapes
:*
dtype0

layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namelayer_normalization/gamma

-layer_normalization/gamma/Read/ReadVariableOpReadVariableOplayer_normalization/gamma*
_output_shapes
:*
dtype0

NoOpNoOp
Λ―
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*―
valueϊ?Bφ? Bξ?
Θ

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer-12
layer-13
layer_with_weights-7
layer-14
layer_with_weights-8
layer-15
layer-16
layer_with_weights-9
layer-17
layer-18
layer_with_weights-10
layer-19
layer_with_weights-11
layer-20
layer-21
layer-22
layer_with_weights-12
layer-23
layer_with_weights-13
layer-24
layer-25
layer_with_weights-14
layer-26
layer-27
layer_with_weights-15
layer-28
layer_with_weights-16
layer-29
layer-30
 layer-31
!layer_with_weights-17
!layer-32
"layer_with_weights-18
"layer-33
#layer-34
$layer_with_weights-19
$layer-35
%layer-36
&layer-37
'layer_with_weights-20
'layer-38
(layer-39
)layer_with_weights-21
)layer-40
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
0_default_save_signature
1	optimizer
2
signatures*
* 
―
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses
9axis
	:gamma
;beta*

<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses
B_attention_axes
C_query_dense
D
_key_dense
E_value_dense
F_softmax
G_dropout_layer
H_output_dense*
₯
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses
O_random_generator* 

P	keras_api* 
―
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses
Waxis
	Xgamma
Ybeta*
Θ
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses

`kernel
abias
 b_jit_compiled_convolution_op*
₯
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses
i_random_generator* 
Θ
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses

pkernel
qbias
 r_jit_compiled_convolution_op*

s	keras_api* 
―
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses
zaxis
	{gamma
|beta*

}	variables
~trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_attention_axes
_query_dense

_key_dense
_value_dense
_softmax
_dropout_layer
_output_dense*
¬
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator* 

	keras_api* 
Έ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis

gamma
	beta*
Ρ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses
‘kernel
	’bias
!£_jit_compiled_convolution_op*
¬
€	variables
₯trainable_variables
¦regularization_losses
§	keras_api
¨__call__
+©&call_and_return_all_conditional_losses
ͺ_random_generator* 
Ρ
«	variables
¬trainable_variables
­regularization_losses
?	keras_api
―__call__
+°&call_and_return_all_conditional_losses
±kernel
	²bias
!³_jit_compiled_convolution_op*

΄	keras_api* 
Έ
΅	variables
Άtrainable_variables
·regularization_losses
Έ	keras_api
Ή__call__
+Ί&call_and_return_all_conditional_losses
	»axis

Όgamma
	½beta*

Ύ	variables
Ώtrainable_variables
ΐregularization_losses
Α	keras_api
Β__call__
+Γ&call_and_return_all_conditional_losses
Δ_attention_axes
Ε_query_dense
Ζ
_key_dense
Η_value_dense
Θ_softmax
Ι_dropout_layer
Κ_output_dense*
¬
Λ	variables
Μtrainable_variables
Νregularization_losses
Ξ	keras_api
Ο__call__
+Π&call_and_return_all_conditional_losses
Ρ_random_generator* 

?	keras_api* 
Έ
Σ	variables
Τtrainable_variables
Υregularization_losses
Φ	keras_api
Χ__call__
+Ψ&call_and_return_all_conditional_losses
	Ωaxis

Ϊgamma
	Ϋbeta*
Ρ
ά	variables
έtrainable_variables
ήregularization_losses
ί	keras_api
ΰ__call__
+α&call_and_return_all_conditional_losses
βkernel
	γbias
!δ_jit_compiled_convolution_op*
¬
ε	variables
ζtrainable_variables
ηregularization_losses
θ	keras_api
ι__call__
+κ&call_and_return_all_conditional_losses
λ_random_generator* 
Ρ
μ	variables
νtrainable_variables
ξregularization_losses
ο	keras_api
π__call__
+ρ&call_and_return_all_conditional_losses
ςkernel
	σbias
!τ_jit_compiled_convolution_op*

υ	keras_api* 
Έ
φ	variables
χtrainable_variables
ψregularization_losses
ω	keras_api
ϊ__call__
+ϋ&call_and_return_all_conditional_losses
	όaxis

ύgamma
	ώbeta*

?	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_attention_axes
_query_dense

_key_dense
_value_dense
_softmax
_dropout_layer
_output_dense*
¬
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator* 

	keras_api* 
Έ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis

gamma
	beta*
Ρ
	variables
trainable_variables
regularization_losses
 	keras_api
‘__call__
+’&call_and_return_all_conditional_losses
£kernel
	€bias
!₯_jit_compiled_convolution_op*
¬
¦	variables
§trainable_variables
¨regularization_losses
©	keras_api
ͺ__call__
+«&call_and_return_all_conditional_losses
¬_random_generator* 
Ρ
­	variables
?trainable_variables
―regularization_losses
°	keras_api
±__call__
+²&call_and_return_all_conditional_losses
³kernel
	΄bias
!΅_jit_compiled_convolution_op*

Ά	keras_api* 

·	variables
Έtrainable_variables
Ήregularization_losses
Ί	keras_api
»__call__
+Ό&call_and_return_all_conditional_losses* 
?
½	variables
Ύtrainable_variables
Ώregularization_losses
ΐ	keras_api
Α__call__
+Β&call_and_return_all_conditional_losses
Γkernel
	Δbias*
¬
Ε	variables
Ζtrainable_variables
Ηregularization_losses
Θ	keras_api
Ι__call__
+Κ&call_and_return_all_conditional_losses
Λ_random_generator* 
?
Μ	variables
Νtrainable_variables
Ξregularization_losses
Ο	keras_api
Π__call__
+Ρ&call_and_return_all_conditional_losses
?kernel
	Σbias*
Τ
:0
;1
Τ2
Υ3
Φ4
Χ5
Ψ6
Ω7
Ϊ8
Ϋ9
X10
Y11
`12
a13
p14
q15
{16
|17
ά18
έ19
ή20
ί21
ΰ22
α23
β24
γ25
26
27
‘28
’29
±30
²31
Ό32
½33
δ34
ε35
ζ36
η37
θ38
ι39
κ40
λ41
Ϊ42
Ϋ43
β44
γ45
ς46
σ47
ύ48
ώ49
μ50
ν51
ξ52
ο53
π54
ρ55
ς56
σ57
58
59
£60
€61
³62
΄63
Γ64
Δ65
?66
Σ67*
Τ
:0
;1
Τ2
Υ3
Φ4
Χ5
Ψ6
Ω7
Ϊ8
Ϋ9
X10
Y11
`12
a13
p14
q15
{16
|17
ά18
έ19
ή20
ί21
ΰ22
α23
β24
γ25
26
27
‘28
’29
±30
²31
Ό32
½33
δ34
ε35
ζ36
η37
θ38
ι39
κ40
λ41
Ϊ42
Ϋ43
β44
γ45
ς46
σ47
ύ48
ώ49
μ50
ν51
ξ52
ο53
π54
ρ55
ς56
σ57
58
59
£60
€61
³62
΄63
Γ64
Δ65
?66
Σ67*
* 
΅
τnon_trainable_variables
υlayers
φmetrics
 χlayer_regularization_losses
ψlayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
0_default_save_signature
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*
:
ωtrace_0
ϊtrace_1
ϋtrace_2
όtrace_3* 
:
ύtrace_0
ώtrace_1
?trace_2
trace_3* 
* 
²
	iter

decay
learning_rate
momentum
rho
:rmsΈ
;rmsΉ
XrmsΊ
Yrms»
`rmsΌ
arms½
prmsΎ
qrmsΏ
{rmsΐ
|rmsΑrmsΒrmsΓ‘rmsΔ’rmsΕ±rmsΖ²rmsΗΌrmsΘ½rmsΙΪrmsΚΫrmsΛβrmsΜγrmsΝςrmsΞσrmsΟύrmsΠώrmsΡrms?rmsΣ£rmsΤ€rmsΥ³rmsΦ΄rmsΧΓrmsΨΔrmsΩ?rmsΪΣrmsΫΤrmsάΥrmsέΦrmsήΧrmsίΨrmsΰΩrmsαΪrmsβΫrmsγάrmsδέrmsεήrmsζίrmsηΰrmsθαrmsιβrmsκγrmsλδrmsμεrmsνζrmsξηrmsοθrmsπιrmsρκrmsςλrmsσμrmsτνrmsυξrmsφοrmsχπrmsψρrmsωςrmsϊσrmsϋ*

serving_default* 

:0
;1*

:0
;1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
hb
VARIABLE_VALUElayer_normalization/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUElayer_normalization/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE*
D
Τ0
Υ1
Φ2
Χ3
Ψ4
Ω5
Ϊ6
Ϋ7*
D
Τ0
Υ1
Φ2
Χ3
Ψ4
Ω5
Ϊ6
Ϋ7*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
α
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
partial_output_shape
full_output_shape
Τkernel
	Υbias*
α
	variables
 trainable_variables
‘regularization_losses
’	keras_api
£__call__
+€&call_and_return_all_conditional_losses
₯partial_output_shape
¦full_output_shape
Φkernel
	Χbias*
α
§	variables
¨trainable_variables
©regularization_losses
ͺ	keras_api
«__call__
+¬&call_and_return_all_conditional_losses
­partial_output_shape
?full_output_shape
Ψkernel
	Ωbias*

―	variables
°trainable_variables
±regularization_losses
²	keras_api
³__call__
+΄&call_and_return_all_conditional_losses* 
¬
΅	variables
Άtrainable_variables
·regularization_losses
Έ	keras_api
Ή__call__
+Ί&call_and_return_all_conditional_losses
»_random_generator* 
α
Ό	variables
½trainable_variables
Ύregularization_losses
Ώ	keras_api
ΐ__call__
+Α&call_and_return_all_conditional_losses
Βpartial_output_shape
Γfull_output_shape
Ϊkernel
	Ϋbias*
* 
* 
* 

Δnon_trainable_variables
Εlayers
Ζmetrics
 Ηlayer_regularization_losses
Θlayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses* 

Ιtrace_0
Κtrace_1* 

Λtrace_0
Μtrace_1* 
* 
* 

X0
Y1*

X0
Y1*
* 

Νnon_trainable_variables
Ξlayers
Οmetrics
 Πlayer_regularization_losses
Ρlayer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses*

?trace_0* 

Σtrace_0* 
* 
jd
VARIABLE_VALUElayer_normalization_1/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUElayer_normalization_1/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE*

`0
a1*

`0
a1*
* 

Τnon_trainable_variables
Υlayers
Φmetrics
 Χlayer_regularization_losses
Ψlayer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses*

Ωtrace_0* 

Ϊtrace_0* 
]W
VARIABLE_VALUEconv1d/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv1d/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

Ϋnon_trainable_variables
άlayers
έmetrics
 ήlayer_regularization_losses
ίlayer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses* 

ΰtrace_0
αtrace_1* 

βtrace_0
γtrace_1* 
* 

p0
q1*

p0
q1*
* 

δnon_trainable_variables
εlayers
ζmetrics
 ηlayer_regularization_losses
θlayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses*

ιtrace_0* 

κtrace_0* 
_Y
VARIABLE_VALUEconv1d_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv1d_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 

{0
|1*

{0
|1*
* 

λnon_trainable_variables
μlayers
νmetrics
 ξlayer_regularization_losses
οlayer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses*

πtrace_0* 

ρtrace_0* 
* 
jd
VARIABLE_VALUElayer_normalization_2/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUElayer_normalization_2/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
D
ά0
έ1
ή2
ί3
ΰ4
α5
β6
γ7*
D
ά0
έ1
ή2
ί3
ΰ4
α5
β6
γ7*
* 

ςnon_trainable_variables
σlayers
τmetrics
 υlayer_regularization_losses
φlayer_metrics
}	variables
~trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

χtrace_0
ψtrace_1* 

ωtrace_0
ϊtrace_1* 
* 
α
ϋ	variables
όtrainable_variables
ύregularization_losses
ώ	keras_api
?__call__
+&call_and_return_all_conditional_losses
partial_output_shape
full_output_shape
άkernel
	έbias*
α
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
partial_output_shape
full_output_shape
ήkernel
	ίbias*
α
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
partial_output_shape
full_output_shape
ΰkernel
	αbias*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
¬
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator* 
α
 	variables
‘trainable_variables
’regularization_losses
£	keras_api
€__call__
+₯&call_and_return_all_conditional_losses
¦partial_output_shape
§full_output_shape
βkernel
	γbias*
* 
* 
* 

¨non_trainable_variables
©layers
ͺmetrics
 «layer_regularization_losses
¬layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

­trace_0
?trace_1* 

―trace_0
°trace_1* 
* 
* 

0
1*

0
1*
* 

±non_trainable_variables
²layers
³metrics
 ΄layer_regularization_losses
΅layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

Άtrace_0* 

·trace_0* 
* 
jd
VARIABLE_VALUElayer_normalization_3/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUElayer_normalization_3/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*

‘0
’1*

‘0
’1*
* 

Έnon_trainable_variables
Ήlayers
Ίmetrics
 »layer_regularization_losses
Όlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses*

½trace_0* 

Ύtrace_0* 
_Y
VARIABLE_VALUEconv1d_2/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv1d_2/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

Ώnon_trainable_variables
ΐlayers
Αmetrics
 Βlayer_regularization_losses
Γlayer_metrics
€	variables
₯trainable_variables
¦regularization_losses
¨__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses* 

Δtrace_0
Εtrace_1* 

Ζtrace_0
Ηtrace_1* 
* 

±0
²1*

±0
²1*
* 

Θnon_trainable_variables
Ιlayers
Κmetrics
 Λlayer_regularization_losses
Μlayer_metrics
«	variables
¬trainable_variables
­regularization_losses
―__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses*

Νtrace_0* 

Ξtrace_0* 
_Y
VARIABLE_VALUEconv1d_3/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv1d_3/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 

Ό0
½1*

Ό0
½1*
* 

Οnon_trainable_variables
Πlayers
Ρmetrics
 ?layer_regularization_losses
Σlayer_metrics
΅	variables
Άtrainable_variables
·regularization_losses
Ή__call__
+Ί&call_and_return_all_conditional_losses
'Ί"call_and_return_conditional_losses*

Τtrace_0* 

Υtrace_0* 
* 
ke
VARIABLE_VALUElayer_normalization_4/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUElayer_normalization_4/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE*
D
δ0
ε1
ζ2
η3
θ4
ι5
κ6
λ7*
D
δ0
ε1
ζ2
η3
θ4
ι5
κ6
λ7*
* 

Φnon_trainable_variables
Χlayers
Ψmetrics
 Ωlayer_regularization_losses
Ϊlayer_metrics
Ύ	variables
Ώtrainable_variables
ΐregularization_losses
Β__call__
+Γ&call_and_return_all_conditional_losses
'Γ"call_and_return_conditional_losses*

Ϋtrace_0
άtrace_1* 

έtrace_0
ήtrace_1* 
* 
α
ί	variables
ΰtrainable_variables
αregularization_losses
β	keras_api
γ__call__
+δ&call_and_return_all_conditional_losses
εpartial_output_shape
ζfull_output_shape
δkernel
	εbias*
α
η	variables
θtrainable_variables
ιregularization_losses
κ	keras_api
λ__call__
+μ&call_and_return_all_conditional_losses
νpartial_output_shape
ξfull_output_shape
ζkernel
	ηbias*
α
ο	variables
πtrainable_variables
ρregularization_losses
ς	keras_api
σ__call__
+τ&call_and_return_all_conditional_losses
υpartial_output_shape
φfull_output_shape
θkernel
	ιbias*

χ	variables
ψtrainable_variables
ωregularization_losses
ϊ	keras_api
ϋ__call__
+ό&call_and_return_all_conditional_losses* 
¬
ύ	variables
ώtrainable_variables
?regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator* 
α
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
partial_output_shape
full_output_shape
κkernel
	λbias*
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Λ	variables
Μtrainable_variables
Νregularization_losses
Ο__call__
+Π&call_and_return_all_conditional_losses
'Π"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 
* 
* 

Ϊ0
Ϋ1*

Ϊ0
Ϋ1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Σ	variables
Τtrainable_variables
Υregularization_losses
Χ__call__
+Ψ&call_and_return_all_conditional_losses
'Ψ"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
ke
VARIABLE_VALUElayer_normalization_5/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUElayer_normalization_5/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE*

β0
γ1*

β0
γ1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
ά	variables
έtrainable_variables
ήregularization_losses
ΰ__call__
+α&call_and_return_all_conditional_losses
'α"call_and_return_conditional_losses*

‘trace_0* 

’trace_0* 
`Z
VARIABLE_VALUEconv1d_4/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv1d_4/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

£non_trainable_variables
€layers
₯metrics
 ¦layer_regularization_losses
§layer_metrics
ε	variables
ζtrainable_variables
ηregularization_losses
ι__call__
+κ&call_and_return_all_conditional_losses
'κ"call_and_return_conditional_losses* 

¨trace_0
©trace_1* 

ͺtrace_0
«trace_1* 
* 

ς0
σ1*

ς0
σ1*
* 

¬non_trainable_variables
­layers
?metrics
 ―layer_regularization_losses
°layer_metrics
μ	variables
νtrainable_variables
ξregularization_losses
π__call__
+ρ&call_and_return_all_conditional_losses
'ρ"call_and_return_conditional_losses*

±trace_0* 

²trace_0* 
`Z
VARIABLE_VALUEconv1d_5/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv1d_5/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 

ύ0
ώ1*

ύ0
ώ1*
* 

³non_trainable_variables
΄layers
΅metrics
 Άlayer_regularization_losses
·layer_metrics
φ	variables
χtrainable_variables
ψregularization_losses
ϊ__call__
+ϋ&call_and_return_all_conditional_losses
'ϋ"call_and_return_conditional_losses*

Έtrace_0* 

Ήtrace_0* 
* 
ke
VARIABLE_VALUElayer_normalization_6/gamma6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUElayer_normalization_6/beta5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUE*
D
μ0
ν1
ξ2
ο3
π4
ρ5
ς6
σ7*
D
μ0
ν1
ξ2
ο3
π4
ρ5
ς6
σ7*
* 

Ίnon_trainable_variables
»layers
Όmetrics
 ½layer_regularization_losses
Ύlayer_metrics
?	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

Ώtrace_0
ΐtrace_1* 

Αtrace_0
Βtrace_1* 
* 
α
Γ	variables
Δtrainable_variables
Εregularization_losses
Ζ	keras_api
Η__call__
+Θ&call_and_return_all_conditional_losses
Ιpartial_output_shape
Κfull_output_shape
μkernel
	νbias*
α
Λ	variables
Μtrainable_variables
Νregularization_losses
Ξ	keras_api
Ο__call__
+Π&call_and_return_all_conditional_losses
Ρpartial_output_shape
?full_output_shape
ξkernel
	οbias*
α
Σ	variables
Τtrainable_variables
Υregularization_losses
Φ	keras_api
Χ__call__
+Ψ&call_and_return_all_conditional_losses
Ωpartial_output_shape
Ϊfull_output_shape
πkernel
	ρbias*

Ϋ	variables
άtrainable_variables
έregularization_losses
ή	keras_api
ί__call__
+ΰ&call_and_return_all_conditional_losses* 
¬
α	variables
βtrainable_variables
γregularization_losses
δ	keras_api
ε__call__
+ζ&call_and_return_all_conditional_losses
η_random_generator* 
α
θ	variables
ιtrainable_variables
κregularization_losses
λ	keras_api
μ__call__
+ν&call_and_return_all_conditional_losses
ξpartial_output_shape
οfull_output_shape
ςkernel
	σbias*
* 
* 
* 

πnon_trainable_variables
ρlayers
ςmetrics
 σlayer_regularization_losses
τlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

υtrace_0
φtrace_1* 

χtrace_0
ψtrace_1* 
* 
* 

0
1*

0
1*
* 

ωnon_trainable_variables
ϊlayers
ϋmetrics
 όlayer_regularization_losses
ύlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

ώtrace_0* 

?trace_0* 
* 
ke
VARIABLE_VALUElayer_normalization_7/gamma6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUElayer_normalization_7/beta5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUE*

£0
€1*

£0
€1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
‘__call__
+’&call_and_return_all_conditional_losses
'’"call_and_return_conditional_losses*

trace_0* 

trace_0* 
`Z
VARIABLE_VALUEconv1d_6/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv1d_6/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¦	variables
§trainable_variables
¨regularization_losses
ͺ__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 
* 

³0
΄1*

³0
΄1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
­	variables
?trainable_variables
―regularization_losses
±__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses*

trace_0* 

trace_0* 
`Z
VARIABLE_VALUEconv1d_7/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv1d_7/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
·	variables
Έtrainable_variables
Ήregularization_losses
»__call__
+Ό&call_and_return_all_conditional_losses
'Ό"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

Γ0
Δ1*

Γ0
Δ1*
* 

non_trainable_variables
layers
 metrics
 ‘layer_regularization_losses
’layer_metrics
½	variables
Ύtrainable_variables
Ώregularization_losses
Α__call__
+Β&call_and_return_all_conditional_losses
'Β"call_and_return_conditional_losses*

£trace_0* 

€trace_0* 
]W
VARIABLE_VALUEdense/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUE
dense/bias5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

₯non_trainable_variables
¦layers
§metrics
 ¨layer_regularization_losses
©layer_metrics
Ε	variables
Ζtrainable_variables
Ηregularization_losses
Ι__call__
+Κ&call_and_return_all_conditional_losses
'Κ"call_and_return_conditional_losses* 

ͺtrace_0
«trace_1* 

¬trace_0
­trace_1* 
* 

?0
Σ1*

?0
Σ1*
* 

?non_trainable_variables
―layers
°metrics
 ±layer_regularization_losses
²layer_metrics
Μ	variables
Νtrainable_variables
Ξregularization_losses
Π__call__
+Ρ&call_and_return_all_conditional_losses
'Ρ"call_and_return_conditional_losses*

³trace_0* 

΄trace_0* 
_Y
VARIABLE_VALUEdense_1/kernel7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_1/bias5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!multi_head_attention/query/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEmulti_head_attention/query/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEmulti_head_attention/key/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEmulti_head_attention/key/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!multi_head_attention/value/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEmulti_head_attention/value/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE,multi_head_attention/attention_output/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUE*multi_head_attention/attention_output/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#multi_head_attention_1/query/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_1/query/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_1/key/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEmulti_head_attention_1/key/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#multi_head_attention_1/value/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_1/value/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE.multi_head_attention_1/attention_output/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE,multi_head_attention_1/attention_output/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#multi_head_attention_2/query/kernel'variables/34/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_2/query/bias'variables/35/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_2/key/kernel'variables/36/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEmulti_head_attention_2/key/bias'variables/37/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#multi_head_attention_2/value/kernel'variables/38/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_2/value/bias'variables/39/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE.multi_head_attention_2/attention_output/kernel'variables/40/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE,multi_head_attention_2/attention_output/bias'variables/41/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#multi_head_attention_3/query/kernel'variables/50/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_3/query/bias'variables/51/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_3/key/kernel'variables/52/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEmulti_head_attention_3/key/bias'variables/53/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#multi_head_attention_3/value/kernel'variables/54/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_3/value/bias'variables/55/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE.multi_head_attention_3/attention_output/kernel'variables/56/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE,multi_head_attention_3/attention_output/bias'variables/57/.ATTRIBUTES/VARIABLE_VALUE*
* 
Β
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40*

΅0
Ά1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
GA
VARIABLE_VALUEiter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEmomentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
E?
VARIABLE_VALUErho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
.
C0
D1
E2
F3
G4
H5*
* 
* 
* 
* 
* 
* 
* 

Τ0
Υ1*

Τ0
Υ1*
* 

·non_trainable_variables
Έlayers
Ήmetrics
 Ίlayer_regularization_losses
»layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 

Φ0
Χ1*

Φ0
Χ1*
* 

Όnon_trainable_variables
½layers
Ύmetrics
 Ώlayer_regularization_losses
ΐlayer_metrics
	variables
 trainable_variables
‘regularization_losses
£__call__
+€&call_and_return_all_conditional_losses
'€"call_and_return_conditional_losses*
* 
* 
* 
* 

Ψ0
Ω1*

Ψ0
Ω1*
* 

Αnon_trainable_variables
Βlayers
Γmetrics
 Δlayer_regularization_losses
Εlayer_metrics
§	variables
¨trainable_variables
©regularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 

Ζnon_trainable_variables
Ηlayers
Θmetrics
 Ιlayer_regularization_losses
Κlayer_metrics
―	variables
°trainable_variables
±regularization_losses
³__call__
+΄&call_and_return_all_conditional_losses
'΄"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

Λnon_trainable_variables
Μlayers
Νmetrics
 Ξlayer_regularization_losses
Οlayer_metrics
΅	variables
Άtrainable_variables
·regularization_losses
Ή__call__
+Ί&call_and_return_all_conditional_losses
'Ί"call_and_return_conditional_losses* 
* 
* 
* 

Ϊ0
Ϋ1*

Ϊ0
Ϋ1*
* 

Πnon_trainable_variables
Ρlayers
?metrics
 Σlayer_regularization_losses
Τlayer_metrics
Ό	variables
½trainable_variables
Ύregularization_losses
ΐ__call__
+Α&call_and_return_all_conditional_losses
'Α"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
4
0
1
2
3
4
5*
* 
* 
* 
* 
* 
* 
* 

ά0
έ1*

ά0
έ1*
* 

Υnon_trainable_variables
Φlayers
Χmetrics
 Ψlayer_regularization_losses
Ωlayer_metrics
ϋ	variables
όtrainable_variables
ύregularization_losses
?__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 

ή0
ί1*

ή0
ί1*
* 

Ϊnon_trainable_variables
Ϋlayers
άmetrics
 έlayer_regularization_losses
ήlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 

ΰ0
α1*

ΰ0
α1*
* 

ίnon_trainable_variables
ΰlayers
αmetrics
 βlayer_regularization_losses
γlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 

δnon_trainable_variables
εlayers
ζmetrics
 ηlayer_regularization_losses
θlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

ιnon_trainable_variables
κlayers
λmetrics
 μlayer_regularization_losses
νlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
* 

β0
γ1*

β0
γ1*
* 

ξnon_trainable_variables
οlayers
πmetrics
 ρlayer_regularization_losses
ςlayer_metrics
 	variables
‘trainable_variables
’regularization_losses
€__call__
+₯&call_and_return_all_conditional_losses
'₯"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
4
Ε0
Ζ1
Η2
Θ3
Ι4
Κ5*
* 
* 
* 
* 
* 
* 
* 

δ0
ε1*

δ0
ε1*
* 

σnon_trainable_variables
τlayers
υmetrics
 φlayer_regularization_losses
χlayer_metrics
ί	variables
ΰtrainable_variables
αregularization_losses
γ__call__
+δ&call_and_return_all_conditional_losses
'δ"call_and_return_conditional_losses*
* 
* 
* 
* 

ζ0
η1*

ζ0
η1*
* 

ψnon_trainable_variables
ωlayers
ϊmetrics
 ϋlayer_regularization_losses
όlayer_metrics
η	variables
θtrainable_variables
ιregularization_losses
λ__call__
+μ&call_and_return_all_conditional_losses
'μ"call_and_return_conditional_losses*
* 
* 
* 
* 

θ0
ι1*

θ0
ι1*
* 

ύnon_trainable_variables
ώlayers
?metrics
 layer_regularization_losses
layer_metrics
ο	variables
πtrainable_variables
ρregularization_losses
σ__call__
+τ&call_and_return_all_conditional_losses
'τ"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
χ	variables
ψtrainable_variables
ωregularization_losses
ϋ__call__
+ό&call_and_return_all_conditional_losses
'ό"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ύ	variables
ώtrainable_variables
?regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
* 

κ0
λ1*

κ0
λ1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
4
0
1
2
3
4
5*
* 
* 
* 
* 
* 
* 
* 

μ0
ν1*

μ0
ν1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Γ	variables
Δtrainable_variables
Εregularization_losses
Η__call__
+Θ&call_and_return_all_conditional_losses
'Θ"call_and_return_conditional_losses*
* 
* 
* 
* 

ξ0
ο1*

ξ0
ο1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Λ	variables
Μtrainable_variables
Νregularization_losses
Ο__call__
+Π&call_and_return_all_conditional_losses
'Π"call_and_return_conditional_losses*
* 
* 
* 
* 

π0
ρ1*

π0
ρ1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Σ	variables
Τtrainable_variables
Υregularization_losses
Χ__call__
+Ψ&call_and_return_all_conditional_losses
'Ψ"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 

 non_trainable_variables
‘layers
’metrics
 £layer_regularization_losses
€layer_metrics
Ϋ	variables
άtrainable_variables
έregularization_losses
ί__call__
+ΰ&call_and_return_all_conditional_losses
'ΰ"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

₯non_trainable_variables
¦layers
§metrics
 ¨layer_regularization_losses
©layer_metrics
α	variables
βtrainable_variables
γregularization_losses
ε__call__
+ζ&call_and_return_all_conditional_losses
'ζ"call_and_return_conditional_losses* 
* 
* 
* 

ς0
σ1*

ς0
σ1*
* 

ͺnon_trainable_variables
«layers
¬metrics
 ­layer_regularization_losses
?layer_metrics
θ	variables
ιtrainable_variables
κregularization_losses
μ__call__
+ν&call_and_return_all_conditional_losses
'ν"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
―	variables
°	keras_api

±total

²count*
M
³	variables
΄	keras_api

΅total

Άcount
·
_fn_kwargs*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

±0
²1*

―	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

΅0
Ά1*

³	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

VARIABLE_VALUElayer_normalization/gamma/rmsSlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization/beta/rmsRlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_1/gamma/rmsSlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_1/beta/rmsRlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEconv1d/kernel/rmsTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEconv1d/bias/rmsRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEconv1d_1/kernel/rmsTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEconv1d_1/bias/rmsRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_2/gamma/rmsSlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_2/beta/rmsRlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_3/gamma/rmsSlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_3/beta/rmsRlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEconv1d_2/kernel/rmsTlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEconv1d_2/bias/rmsRlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEconv1d_3/kernel/rmsTlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEconv1d_3/bias/rmsRlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_4/gamma/rmsTlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_4/beta/rmsSlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_5/gamma/rmsTlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_5/beta/rmsSlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEconv1d_4/kernel/rmsUlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEconv1d_4/bias/rmsSlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEconv1d_5/kernel/rmsUlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEconv1d_5/bias/rmsSlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_6/gamma/rmsTlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_6/beta/rmsSlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_7/gamma/rmsTlayer_with_weights-17/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_7/beta/rmsSlayer_with_weights-17/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEconv1d_6/kernel/rmsUlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEconv1d_6/bias/rmsSlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEconv1d_7/kernel/rmsUlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEconv1d_7/bias/rmsSlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEdense/kernel/rmsUlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEdense/bias/rmsSlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEdense_1/kernel/rmsUlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEdense_1/bias/rmsSlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUE%multi_head_attention/query/kernel/rmsDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE#multi_head_attention/query/bias/rmsDvariables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE#multi_head_attention/key/kernel/rmsDvariables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE!multi_head_attention/key/bias/rmsDvariables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUE%multi_head_attention/value/kernel/rmsDvariables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE#multi_head_attention/value/bias/rmsDvariables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE0multi_head_attention/attention_output/kernel/rmsDvariables/8/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE.multi_head_attention/attention_output/bias/rmsDvariables/9/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE'multi_head_attention_1/query/kernel/rmsEvariables/18/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE%multi_head_attention_1/query/bias/rmsEvariables/19/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE%multi_head_attention_1/key/kernel/rmsEvariables/20/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE#multi_head_attention_1/key/bias/rmsEvariables/21/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE'multi_head_attention_1/value/kernel/rmsEvariables/22/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE%multi_head_attention_1/value/bias/rmsEvariables/23/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE2multi_head_attention_1/attention_output/kernel/rmsEvariables/24/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE0multi_head_attention_1/attention_output/bias/rmsEvariables/25/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE'multi_head_attention_2/query/kernel/rmsEvariables/34/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE%multi_head_attention_2/query/bias/rmsEvariables/35/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE%multi_head_attention_2/key/kernel/rmsEvariables/36/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE#multi_head_attention_2/key/bias/rmsEvariables/37/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE'multi_head_attention_2/value/kernel/rmsEvariables/38/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE%multi_head_attention_2/value/bias/rmsEvariables/39/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE2multi_head_attention_2/attention_output/kernel/rmsEvariables/40/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE0multi_head_attention_2/attention_output/bias/rmsEvariables/41/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE'multi_head_attention_3/query/kernel/rmsEvariables/50/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE%multi_head_attention_3/query/bias/rmsEvariables/51/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE%multi_head_attention_3/key/kernel/rmsEvariables/52/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE#multi_head_attention_3/key/bias/rmsEvariables/53/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE'multi_head_attention_3/value/kernel/rmsEvariables/54/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE%multi_head_attention_3/value/bias/rmsEvariables/55/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE2multi_head_attention_3/attention_output/kernel/rmsEvariables/56/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE0multi_head_attention_3/attention_output/bias/rmsEvariables/57/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_input_1Placeholder*,
_output_shapes
:?????????Θ*
dtype0*!
shape:?????????Θ
ι
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1layer_normalization/gammalayer_normalization/beta!multi_head_attention/query/kernelmulti_head_attention/query/biasmulti_head_attention/key/kernelmulti_head_attention/key/bias!multi_head_attention/value/kernelmulti_head_attention/value/bias,multi_head_attention/attention_output/kernel*multi_head_attention/attention_output/biaslayer_normalization_1/gammalayer_normalization_1/betaconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biaslayer_normalization_2/gammalayer_normalization_2/beta#multi_head_attention_1/query/kernel!multi_head_attention_1/query/bias!multi_head_attention_1/key/kernelmulti_head_attention_1/key/bias#multi_head_attention_1/value/kernel!multi_head_attention_1/value/bias.multi_head_attention_1/attention_output/kernel,multi_head_attention_1/attention_output/biaslayer_normalization_3/gammalayer_normalization_3/betaconv1d_2/kernelconv1d_2/biasconv1d_3/kernelconv1d_3/biaslayer_normalization_4/gammalayer_normalization_4/beta#multi_head_attention_2/query/kernel!multi_head_attention_2/query/bias!multi_head_attention_2/key/kernelmulti_head_attention_2/key/bias#multi_head_attention_2/value/kernel!multi_head_attention_2/value/bias.multi_head_attention_2/attention_output/kernel,multi_head_attention_2/attention_output/biaslayer_normalization_5/gammalayer_normalization_5/betaconv1d_4/kernelconv1d_4/biasconv1d_5/kernelconv1d_5/biaslayer_normalization_6/gammalayer_normalization_6/beta#multi_head_attention_3/query/kernel!multi_head_attention_3/query/bias!multi_head_attention_3/key/kernelmulti_head_attention_3/key/bias#multi_head_attention_3/value/kernel!multi_head_attention_3/value/bias.multi_head_attention_3/attention_output/kernel,multi_head_attention_3/attention_output/biaslayer_normalization_7/gammalayer_normalization_7/betaconv1d_6/kernelconv1d_6/biasconv1d_7/kernelconv1d_7/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias*P
TinI
G2E*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*f
_read_only_resource_inputsH
FD	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCD*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_16190
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ύ;
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-layer_normalization/gamma/Read/ReadVariableOp,layer_normalization/beta/Read/ReadVariableOp/layer_normalization_1/gamma/Read/ReadVariableOp.layer_normalization_1/beta/Read/ReadVariableOp!conv1d/kernel/Read/ReadVariableOpconv1d/bias/Read/ReadVariableOp#conv1d_1/kernel/Read/ReadVariableOp!conv1d_1/bias/Read/ReadVariableOp/layer_normalization_2/gamma/Read/ReadVariableOp.layer_normalization_2/beta/Read/ReadVariableOp/layer_normalization_3/gamma/Read/ReadVariableOp.layer_normalization_3/beta/Read/ReadVariableOp#conv1d_2/kernel/Read/ReadVariableOp!conv1d_2/bias/Read/ReadVariableOp#conv1d_3/kernel/Read/ReadVariableOp!conv1d_3/bias/Read/ReadVariableOp/layer_normalization_4/gamma/Read/ReadVariableOp.layer_normalization_4/beta/Read/ReadVariableOp/layer_normalization_5/gamma/Read/ReadVariableOp.layer_normalization_5/beta/Read/ReadVariableOp#conv1d_4/kernel/Read/ReadVariableOp!conv1d_4/bias/Read/ReadVariableOp#conv1d_5/kernel/Read/ReadVariableOp!conv1d_5/bias/Read/ReadVariableOp/layer_normalization_6/gamma/Read/ReadVariableOp.layer_normalization_6/beta/Read/ReadVariableOp/layer_normalization_7/gamma/Read/ReadVariableOp.layer_normalization_7/beta/Read/ReadVariableOp#conv1d_6/kernel/Read/ReadVariableOp!conv1d_6/bias/Read/ReadVariableOp#conv1d_7/kernel/Read/ReadVariableOp!conv1d_7/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp5multi_head_attention/query/kernel/Read/ReadVariableOp3multi_head_attention/query/bias/Read/ReadVariableOp3multi_head_attention/key/kernel/Read/ReadVariableOp1multi_head_attention/key/bias/Read/ReadVariableOp5multi_head_attention/value/kernel/Read/ReadVariableOp3multi_head_attention/value/bias/Read/ReadVariableOp@multi_head_attention/attention_output/kernel/Read/ReadVariableOp>multi_head_attention/attention_output/bias/Read/ReadVariableOp7multi_head_attention_1/query/kernel/Read/ReadVariableOp5multi_head_attention_1/query/bias/Read/ReadVariableOp5multi_head_attention_1/key/kernel/Read/ReadVariableOp3multi_head_attention_1/key/bias/Read/ReadVariableOp7multi_head_attention_1/value/kernel/Read/ReadVariableOp5multi_head_attention_1/value/bias/Read/ReadVariableOpBmulti_head_attention_1/attention_output/kernel/Read/ReadVariableOp@multi_head_attention_1/attention_output/bias/Read/ReadVariableOp7multi_head_attention_2/query/kernel/Read/ReadVariableOp5multi_head_attention_2/query/bias/Read/ReadVariableOp5multi_head_attention_2/key/kernel/Read/ReadVariableOp3multi_head_attention_2/key/bias/Read/ReadVariableOp7multi_head_attention_2/value/kernel/Read/ReadVariableOp5multi_head_attention_2/value/bias/Read/ReadVariableOpBmulti_head_attention_2/attention_output/kernel/Read/ReadVariableOp@multi_head_attention_2/attention_output/bias/Read/ReadVariableOp7multi_head_attention_3/query/kernel/Read/ReadVariableOp5multi_head_attention_3/query/bias/Read/ReadVariableOp5multi_head_attention_3/key/kernel/Read/ReadVariableOp3multi_head_attention_3/key/bias/Read/ReadVariableOp7multi_head_attention_3/value/kernel/Read/ReadVariableOp5multi_head_attention_3/value/bias/Read/ReadVariableOpBmulti_head_attention_3/attention_output/kernel/Read/ReadVariableOp@multi_head_attention_3/attention_output/bias/Read/ReadVariableOpiter/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpmomentum/Read/ReadVariableOprho/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp1layer_normalization/gamma/rms/Read/ReadVariableOp0layer_normalization/beta/rms/Read/ReadVariableOp3layer_normalization_1/gamma/rms/Read/ReadVariableOp2layer_normalization_1/beta/rms/Read/ReadVariableOp%conv1d/kernel/rms/Read/ReadVariableOp#conv1d/bias/rms/Read/ReadVariableOp'conv1d_1/kernel/rms/Read/ReadVariableOp%conv1d_1/bias/rms/Read/ReadVariableOp3layer_normalization_2/gamma/rms/Read/ReadVariableOp2layer_normalization_2/beta/rms/Read/ReadVariableOp3layer_normalization_3/gamma/rms/Read/ReadVariableOp2layer_normalization_3/beta/rms/Read/ReadVariableOp'conv1d_2/kernel/rms/Read/ReadVariableOp%conv1d_2/bias/rms/Read/ReadVariableOp'conv1d_3/kernel/rms/Read/ReadVariableOp%conv1d_3/bias/rms/Read/ReadVariableOp3layer_normalization_4/gamma/rms/Read/ReadVariableOp2layer_normalization_4/beta/rms/Read/ReadVariableOp3layer_normalization_5/gamma/rms/Read/ReadVariableOp2layer_normalization_5/beta/rms/Read/ReadVariableOp'conv1d_4/kernel/rms/Read/ReadVariableOp%conv1d_4/bias/rms/Read/ReadVariableOp'conv1d_5/kernel/rms/Read/ReadVariableOp%conv1d_5/bias/rms/Read/ReadVariableOp3layer_normalization_6/gamma/rms/Read/ReadVariableOp2layer_normalization_6/beta/rms/Read/ReadVariableOp3layer_normalization_7/gamma/rms/Read/ReadVariableOp2layer_normalization_7/beta/rms/Read/ReadVariableOp'conv1d_6/kernel/rms/Read/ReadVariableOp%conv1d_6/bias/rms/Read/ReadVariableOp'conv1d_7/kernel/rms/Read/ReadVariableOp%conv1d_7/bias/rms/Read/ReadVariableOp$dense/kernel/rms/Read/ReadVariableOp"dense/bias/rms/Read/ReadVariableOp&dense_1/kernel/rms/Read/ReadVariableOp$dense_1/bias/rms/Read/ReadVariableOp9multi_head_attention/query/kernel/rms/Read/ReadVariableOp7multi_head_attention/query/bias/rms/Read/ReadVariableOp7multi_head_attention/key/kernel/rms/Read/ReadVariableOp5multi_head_attention/key/bias/rms/Read/ReadVariableOp9multi_head_attention/value/kernel/rms/Read/ReadVariableOp7multi_head_attention/value/bias/rms/Read/ReadVariableOpDmulti_head_attention/attention_output/kernel/rms/Read/ReadVariableOpBmulti_head_attention/attention_output/bias/rms/Read/ReadVariableOp;multi_head_attention_1/query/kernel/rms/Read/ReadVariableOp9multi_head_attention_1/query/bias/rms/Read/ReadVariableOp9multi_head_attention_1/key/kernel/rms/Read/ReadVariableOp7multi_head_attention_1/key/bias/rms/Read/ReadVariableOp;multi_head_attention_1/value/kernel/rms/Read/ReadVariableOp9multi_head_attention_1/value/bias/rms/Read/ReadVariableOpFmulti_head_attention_1/attention_output/kernel/rms/Read/ReadVariableOpDmulti_head_attention_1/attention_output/bias/rms/Read/ReadVariableOp;multi_head_attention_2/query/kernel/rms/Read/ReadVariableOp9multi_head_attention_2/query/bias/rms/Read/ReadVariableOp9multi_head_attention_2/key/kernel/rms/Read/ReadVariableOp7multi_head_attention_2/key/bias/rms/Read/ReadVariableOp;multi_head_attention_2/value/kernel/rms/Read/ReadVariableOp9multi_head_attention_2/value/bias/rms/Read/ReadVariableOpFmulti_head_attention_2/attention_output/kernel/rms/Read/ReadVariableOpDmulti_head_attention_2/attention_output/bias/rms/Read/ReadVariableOp;multi_head_attention_3/query/kernel/rms/Read/ReadVariableOp9multi_head_attention_3/query/bias/rms/Read/ReadVariableOp9multi_head_attention_3/key/kernel/rms/Read/ReadVariableOp7multi_head_attention_3/key/bias/rms/Read/ReadVariableOp;multi_head_attention_3/value/kernel/rms/Read/ReadVariableOp9multi_head_attention_3/value/bias/rms/Read/ReadVariableOpFmulti_head_attention_3/attention_output/kernel/rms/Read/ReadVariableOpDmulti_head_attention_3/attention_output/bias/rms/Read/ReadVariableOpConst*‘
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_19026
ε$
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer_normalization/gammalayer_normalization/betalayer_normalization_1/gammalayer_normalization_1/betaconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biaslayer_normalization_2/gammalayer_normalization_2/betalayer_normalization_3/gammalayer_normalization_3/betaconv1d_2/kernelconv1d_2/biasconv1d_3/kernelconv1d_3/biaslayer_normalization_4/gammalayer_normalization_4/betalayer_normalization_5/gammalayer_normalization_5/betaconv1d_4/kernelconv1d_4/biasconv1d_5/kernelconv1d_5/biaslayer_normalization_6/gammalayer_normalization_6/betalayer_normalization_7/gammalayer_normalization_7/betaconv1d_6/kernelconv1d_6/biasconv1d_7/kernelconv1d_7/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias!multi_head_attention/query/kernelmulti_head_attention/query/biasmulti_head_attention/key/kernelmulti_head_attention/key/bias!multi_head_attention/value/kernelmulti_head_attention/value/bias,multi_head_attention/attention_output/kernel*multi_head_attention/attention_output/bias#multi_head_attention_1/query/kernel!multi_head_attention_1/query/bias!multi_head_attention_1/key/kernelmulti_head_attention_1/key/bias#multi_head_attention_1/value/kernel!multi_head_attention_1/value/bias.multi_head_attention_1/attention_output/kernel,multi_head_attention_1/attention_output/bias#multi_head_attention_2/query/kernel!multi_head_attention_2/query/bias!multi_head_attention_2/key/kernelmulti_head_attention_2/key/bias#multi_head_attention_2/value/kernel!multi_head_attention_2/value/bias.multi_head_attention_2/attention_output/kernel,multi_head_attention_2/attention_output/bias#multi_head_attention_3/query/kernel!multi_head_attention_3/query/bias!multi_head_attention_3/key/kernelmulti_head_attention_3/key/bias#multi_head_attention_3/value/kernel!multi_head_attention_3/value/bias.multi_head_attention_3/attention_output/kernel,multi_head_attention_3/attention_output/biasiterdecaylearning_ratemomentumrhototal_1count_1totalcountlayer_normalization/gamma/rmslayer_normalization/beta/rmslayer_normalization_1/gamma/rmslayer_normalization_1/beta/rmsconv1d/kernel/rmsconv1d/bias/rmsconv1d_1/kernel/rmsconv1d_1/bias/rmslayer_normalization_2/gamma/rmslayer_normalization_2/beta/rmslayer_normalization_3/gamma/rmslayer_normalization_3/beta/rmsconv1d_2/kernel/rmsconv1d_2/bias/rmsconv1d_3/kernel/rmsconv1d_3/bias/rmslayer_normalization_4/gamma/rmslayer_normalization_4/beta/rmslayer_normalization_5/gamma/rmslayer_normalization_5/beta/rmsconv1d_4/kernel/rmsconv1d_4/bias/rmsconv1d_5/kernel/rmsconv1d_5/bias/rmslayer_normalization_6/gamma/rmslayer_normalization_6/beta/rmslayer_normalization_7/gamma/rmslayer_normalization_7/beta/rmsconv1d_6/kernel/rmsconv1d_6/bias/rmsconv1d_7/kernel/rmsconv1d_7/bias/rmsdense/kernel/rmsdense/bias/rmsdense_1/kernel/rmsdense_1/bias/rms%multi_head_attention/query/kernel/rms#multi_head_attention/query/bias/rms#multi_head_attention/key/kernel/rms!multi_head_attention/key/bias/rms%multi_head_attention/value/kernel/rms#multi_head_attention/value/bias/rms0multi_head_attention/attention_output/kernel/rms.multi_head_attention/attention_output/bias/rms'multi_head_attention_1/query/kernel/rms%multi_head_attention_1/query/bias/rms%multi_head_attention_1/key/kernel/rms#multi_head_attention_1/key/bias/rms'multi_head_attention_1/value/kernel/rms%multi_head_attention_1/value/bias/rms2multi_head_attention_1/attention_output/kernel/rms0multi_head_attention_1/attention_output/bias/rms'multi_head_attention_2/query/kernel/rms%multi_head_attention_2/query/bias/rms%multi_head_attention_2/key/kernel/rms#multi_head_attention_2/key/bias/rms'multi_head_attention_2/value/kernel/rms%multi_head_attention_2/value/bias/rms2multi_head_attention_2/attention_output/kernel/rms0multi_head_attention_2/attention_output/bias/rms'multi_head_attention_3/query/kernel/rms%multi_head_attention_3/query/bias/rms%multi_head_attention_3/key/kernel/rms#multi_head_attention_3/key/bias/rms'multi_head_attention_3/value/kernel/rms%multi_head_attention_3/value/bias/rms2multi_head_attention_3/attention_output/kernel/rms0multi_head_attention_3/attention_output/bias/rms* 
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_19471ξ²,
ύ

P__inference_layer_normalization_7_layer_call_and_return_conditional_losses_14105

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(r
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:?????????Θ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¬
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θb
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:?????????Θ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θh
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θw
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????Θw
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Θg
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:?????????Θ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
­
C
'__inference_dropout_layer_call_fn_17504

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_13576e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:?????????Θ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
λ
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_17883

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:?????????Θ`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:?????????Θ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
Ι	
τ
B__inference_dense_1_layer_call_and_return_conditional_losses_18568

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
Ϋ
b
D__inference_dropout_8_layer_call_and_return_conditional_losses_18537

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:?????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
΅*
ψ
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_17457	
query	
valueA
+query_einsum_einsum_readvariableop_resource:23
!query_add_readvariableop_resource:2?
)key_einsum_einsum_readvariableop_resource:21
key_add_readvariableop_resource:2A
+value_einsum_einsum_readvariableop_resource:23
!value_add_readvariableop_resource:2L
6attention_output_einsum_einsum_readvariableop_resource:2:
,attention_output_add_readvariableop_resource:
identity’#attention_output/add/ReadVariableOp’-attention_output/einsum/Einsum/ReadVariableOp’key/add/ReadVariableOp’ key/einsum/Einsum/ReadVariableOp’query/add/ReadVariableOp’"query/einsum/Einsum/ReadVariableOp’value/add/ReadVariableOp’"value/einsum/Einsum/ReadVariableOp
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:2*
dtype0
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0­
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:2*
dtype0
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:2*
dtype0
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΓΠ>d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:?????????Θ2
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:?????????ΘΘ*
equationaecd,abcd->acben
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:?????????ΘΘu
dropout_9/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*1
_output_shapes
:?????????ΘΘ¨
einsum_1/EinsumEinsumdropout_9/Identity:output:0value/add:z:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationacbe,aecd->abcd¨
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0Φ
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????Θ*
equationabcd,cde->abe
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0ͺ
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θl
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:?????????ΘΨ
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Θ:?????????Θ: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:?????????Θ

_user_specified_namequery:SO
,
_output_shapes
:?????????Θ

_user_specified_namevalue
Ι	
τ
B__inference_dense_1_layer_call_and_return_conditional_losses_14197

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs

Ά
%__inference_model_layer_call_fn_14343
input_1
unknown:
	unknown_0:
	unknown_1:2
	unknown_2:2
	unknown_3:2
	unknown_4:2
	unknown_5:2
	unknown_6:2
	unknown_7:2
	unknown_8:
	unknown_9:

unknown_10: 

unknown_11:

unknown_12: 

unknown_13:

unknown_14:

unknown_15:

unknown_16: 

unknown_17:2

unknown_18:2 

unknown_19:2

unknown_20:2 

unknown_21:2

unknown_22:2 

unknown_23:2

unknown_24:

unknown_25:

unknown_26: 

unknown_27:

unknown_28: 

unknown_29:

unknown_30:

unknown_31:

unknown_32: 

unknown_33:2

unknown_34:2 

unknown_35:2

unknown_36:2 

unknown_37:2

unknown_38:2 

unknown_39:2

unknown_40:

unknown_41:

unknown_42: 

unknown_43:

unknown_44: 

unknown_45:

unknown_46:

unknown_47:

unknown_48: 

unknown_49:2

unknown_50:2 

unknown_51:2

unknown_52:2 

unknown_53:2

unknown_54:2 

unknown_55:2

unknown_56:

unknown_57:

unknown_58: 

unknown_59:

unknown_60: 

unknown_61:

unknown_62:

unknown_63:
Θ

unknown_64:	

unknown_65:	

unknown_66:
identity’StatefulPartitionedCallι	
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66*P
TinI
G2E*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*f
_read_only_resource_inputsH
FD	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCD*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_14204o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*΅
_input_shapes£
 :?????????Θ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:?????????Θ
!
_user_specified_name	input_1
δύ
ΐI
 __inference__wrapped_model_13470
input_1M
?model_layer_normalization_batchnorm_mul_readvariableop_resource:I
;model_layer_normalization_batchnorm_readvariableop_resource:\
Fmodel_multi_head_attention_query_einsum_einsum_readvariableop_resource:2N
<model_multi_head_attention_query_add_readvariableop_resource:2Z
Dmodel_multi_head_attention_key_einsum_einsum_readvariableop_resource:2L
:model_multi_head_attention_key_add_readvariableop_resource:2\
Fmodel_multi_head_attention_value_einsum_einsum_readvariableop_resource:2N
<model_multi_head_attention_value_add_readvariableop_resource:2g
Qmodel_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:2U
Gmodel_multi_head_attention_attention_output_add_readvariableop_resource:O
Amodel_layer_normalization_1_batchnorm_mul_readvariableop_resource:K
=model_layer_normalization_1_batchnorm_readvariableop_resource:N
8model_conv1d_conv1d_expanddims_1_readvariableop_resource::
,model_conv1d_biasadd_readvariableop_resource:P
:model_conv1d_1_conv1d_expanddims_1_readvariableop_resource:<
.model_conv1d_1_biasadd_readvariableop_resource:O
Amodel_layer_normalization_2_batchnorm_mul_readvariableop_resource:K
=model_layer_normalization_2_batchnorm_readvariableop_resource:^
Hmodel_multi_head_attention_1_query_einsum_einsum_readvariableop_resource:2P
>model_multi_head_attention_1_query_add_readvariableop_resource:2\
Fmodel_multi_head_attention_1_key_einsum_einsum_readvariableop_resource:2N
<model_multi_head_attention_1_key_add_readvariableop_resource:2^
Hmodel_multi_head_attention_1_value_einsum_einsum_readvariableop_resource:2P
>model_multi_head_attention_1_value_add_readvariableop_resource:2i
Smodel_multi_head_attention_1_attention_output_einsum_einsum_readvariableop_resource:2W
Imodel_multi_head_attention_1_attention_output_add_readvariableop_resource:O
Amodel_layer_normalization_3_batchnorm_mul_readvariableop_resource:K
=model_layer_normalization_3_batchnorm_readvariableop_resource:P
:model_conv1d_2_conv1d_expanddims_1_readvariableop_resource:<
.model_conv1d_2_biasadd_readvariableop_resource:P
:model_conv1d_3_conv1d_expanddims_1_readvariableop_resource:<
.model_conv1d_3_biasadd_readvariableop_resource:O
Amodel_layer_normalization_4_batchnorm_mul_readvariableop_resource:K
=model_layer_normalization_4_batchnorm_readvariableop_resource:^
Hmodel_multi_head_attention_2_query_einsum_einsum_readvariableop_resource:2P
>model_multi_head_attention_2_query_add_readvariableop_resource:2\
Fmodel_multi_head_attention_2_key_einsum_einsum_readvariableop_resource:2N
<model_multi_head_attention_2_key_add_readvariableop_resource:2^
Hmodel_multi_head_attention_2_value_einsum_einsum_readvariableop_resource:2P
>model_multi_head_attention_2_value_add_readvariableop_resource:2i
Smodel_multi_head_attention_2_attention_output_einsum_einsum_readvariableop_resource:2W
Imodel_multi_head_attention_2_attention_output_add_readvariableop_resource:O
Amodel_layer_normalization_5_batchnorm_mul_readvariableop_resource:K
=model_layer_normalization_5_batchnorm_readvariableop_resource:P
:model_conv1d_4_conv1d_expanddims_1_readvariableop_resource:<
.model_conv1d_4_biasadd_readvariableop_resource:P
:model_conv1d_5_conv1d_expanddims_1_readvariableop_resource:<
.model_conv1d_5_biasadd_readvariableop_resource:O
Amodel_layer_normalization_6_batchnorm_mul_readvariableop_resource:K
=model_layer_normalization_6_batchnorm_readvariableop_resource:^
Hmodel_multi_head_attention_3_query_einsum_einsum_readvariableop_resource:2P
>model_multi_head_attention_3_query_add_readvariableop_resource:2\
Fmodel_multi_head_attention_3_key_einsum_einsum_readvariableop_resource:2N
<model_multi_head_attention_3_key_add_readvariableop_resource:2^
Hmodel_multi_head_attention_3_value_einsum_einsum_readvariableop_resource:2P
>model_multi_head_attention_3_value_add_readvariableop_resource:2i
Smodel_multi_head_attention_3_attention_output_einsum_einsum_readvariableop_resource:2W
Imodel_multi_head_attention_3_attention_output_add_readvariableop_resource:O
Amodel_layer_normalization_7_batchnorm_mul_readvariableop_resource:K
=model_layer_normalization_7_batchnorm_readvariableop_resource:P
:model_conv1d_6_conv1d_expanddims_1_readvariableop_resource:<
.model_conv1d_6_biasadd_readvariableop_resource:P
:model_conv1d_7_conv1d_expanddims_1_readvariableop_resource:<
.model_conv1d_7_biasadd_readvariableop_resource:>
*model_dense_matmul_readvariableop_resource:
Θ:
+model_dense_biasadd_readvariableop_resource:	?
,model_dense_1_matmul_readvariableop_resource:	;
-model_dense_1_biasadd_readvariableop_resource:
identity’#model/conv1d/BiasAdd/ReadVariableOp’/model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp’%model/conv1d_1/BiasAdd/ReadVariableOp’1model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp’%model/conv1d_2/BiasAdd/ReadVariableOp’1model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp’%model/conv1d_3/BiasAdd/ReadVariableOp’1model/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp’%model/conv1d_4/BiasAdd/ReadVariableOp’1model/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp’%model/conv1d_5/BiasAdd/ReadVariableOp’1model/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp’%model/conv1d_6/BiasAdd/ReadVariableOp’1model/conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp’%model/conv1d_7/BiasAdd/ReadVariableOp’1model/conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp’"model/dense/BiasAdd/ReadVariableOp’!model/dense/MatMul/ReadVariableOp’$model/dense_1/BiasAdd/ReadVariableOp’#model/dense_1/MatMul/ReadVariableOp’2model/layer_normalization/batchnorm/ReadVariableOp’6model/layer_normalization/batchnorm/mul/ReadVariableOp’4model/layer_normalization_1/batchnorm/ReadVariableOp’8model/layer_normalization_1/batchnorm/mul/ReadVariableOp’4model/layer_normalization_2/batchnorm/ReadVariableOp’8model/layer_normalization_2/batchnorm/mul/ReadVariableOp’4model/layer_normalization_3/batchnorm/ReadVariableOp’8model/layer_normalization_3/batchnorm/mul/ReadVariableOp’4model/layer_normalization_4/batchnorm/ReadVariableOp’8model/layer_normalization_4/batchnorm/mul/ReadVariableOp’4model/layer_normalization_5/batchnorm/ReadVariableOp’8model/layer_normalization_5/batchnorm/mul/ReadVariableOp’4model/layer_normalization_6/batchnorm/ReadVariableOp’8model/layer_normalization_6/batchnorm/mul/ReadVariableOp’4model/layer_normalization_7/batchnorm/ReadVariableOp’8model/layer_normalization_7/batchnorm/mul/ReadVariableOp’>model/multi_head_attention/attention_output/add/ReadVariableOp’Hmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp’1model/multi_head_attention/key/add/ReadVariableOp’;model/multi_head_attention/key/einsum/Einsum/ReadVariableOp’3model/multi_head_attention/query/add/ReadVariableOp’=model/multi_head_attention/query/einsum/Einsum/ReadVariableOp’3model/multi_head_attention/value/add/ReadVariableOp’=model/multi_head_attention/value/einsum/Einsum/ReadVariableOp’@model/multi_head_attention_1/attention_output/add/ReadVariableOp’Jmodel/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp’3model/multi_head_attention_1/key/add/ReadVariableOp’=model/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp’5model/multi_head_attention_1/query/add/ReadVariableOp’?model/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp’5model/multi_head_attention_1/value/add/ReadVariableOp’?model/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp’@model/multi_head_attention_2/attention_output/add/ReadVariableOp’Jmodel/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp’3model/multi_head_attention_2/key/add/ReadVariableOp’=model/multi_head_attention_2/key/einsum/Einsum/ReadVariableOp’5model/multi_head_attention_2/query/add/ReadVariableOp’?model/multi_head_attention_2/query/einsum/Einsum/ReadVariableOp’5model/multi_head_attention_2/value/add/ReadVariableOp’?model/multi_head_attention_2/value/einsum/Einsum/ReadVariableOp’@model/multi_head_attention_3/attention_output/add/ReadVariableOp’Jmodel/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp’3model/multi_head_attention_3/key/add/ReadVariableOp’=model/multi_head_attention_3/key/einsum/Einsum/ReadVariableOp’5model/multi_head_attention_3/query/add/ReadVariableOp’?model/multi_head_attention_3/query/einsum/Einsum/ReadVariableOp’5model/multi_head_attention_3/value/add/ReadVariableOp’?model/multi_head_attention_3/value/einsum/Einsum/ReadVariableOp
8model/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Β
&model/layer_normalization/moments/meanMeaninput_1Amodel/layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(¦
.model/layer_normalization/moments/StopGradientStopGradient/model/layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:?????????ΘΑ
3model/layer_normalization/moments/SquaredDifferenceSquaredDifferenceinput_17model/layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θ
<model/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ϊ
*model/layer_normalization/moments/varianceMean7model/layer_normalization/moments/SquaredDifference:z:0Emodel/layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(n
)model/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Π
'model/layer_normalization/batchnorm/addAddV23model/layer_normalization/moments/variance:output:02model/layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θ
)model/layer_normalization/batchnorm/RsqrtRsqrt+model/layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:?????????Θ²
6model/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp?model_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Τ
'model/layer_normalization/batchnorm/mulMul-model/layer_normalization/batchnorm/Rsqrt:y:0>model/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ
)model/layer_normalization/batchnorm/mul_1Mulinput_1+model/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ΘΕ
)model/layer_normalization/batchnorm/mul_2Mul/model/layer_normalization/moments/mean:output:0+model/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θͺ
2model/layer_normalization/batchnorm/ReadVariableOpReadVariableOp;model_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Π
'model/layer_normalization/batchnorm/subSub:model/layer_normalization/batchnorm/ReadVariableOp:value:0-model/layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????ΘΕ
)model/layer_normalization/batchnorm/add_1AddV2-model/layer_normalization/batchnorm/mul_1:z:0+model/layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????ΘΘ
=model/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpFmodel_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0
.model/multi_head_attention/query/einsum/EinsumEinsum-model/layer_normalization/batchnorm/add_1:z:0Emodel/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde°
3model/multi_head_attention/query/add/ReadVariableOpReadVariableOp<model_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:2*
dtype0ή
$model/multi_head_attention/query/addAddV27model/multi_head_attention/query/einsum/Einsum:output:0;model/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2Δ
;model/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpDmodel_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0
,model/multi_head_attention/key/einsum/EinsumEinsum-model/layer_normalization/batchnorm/add_1:z:0Cmodel/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde¬
1model/multi_head_attention/key/add/ReadVariableOpReadVariableOp:model_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:2*
dtype0Ψ
"model/multi_head_attention/key/addAddV25model/multi_head_attention/key/einsum/Einsum:output:09model/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2Θ
=model/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpFmodel_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0
.model/multi_head_attention/value/einsum/EinsumEinsum-model/layer_normalization/batchnorm/add_1:z:0Emodel/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde°
3model/multi_head_attention/value/add/ReadVariableOpReadVariableOp<model_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:2*
dtype0ή
$model/multi_head_attention/value/addAddV27model/multi_head_attention/value/einsum/Einsum:output:0;model/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2e
 model/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΓΠ>΅
model/multi_head_attention/MulMul(model/multi_head_attention/query/add:z:0)model/multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:?????????Θ2β
(model/multi_head_attention/einsum/EinsumEinsum&model/multi_head_attention/key/add:z:0"model/multi_head_attention/Mul:z:0*
N*
T0*1
_output_shapes
:?????????ΘΘ*
equationaecd,abcd->acbe€
*model/multi_head_attention/softmax/SoftmaxSoftmax1model/multi_head_attention/einsum/Einsum:output:0*
T0*1
_output_shapes
:?????????ΘΘ«
-model/multi_head_attention/dropout_9/IdentityIdentity4model/multi_head_attention/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:?????????ΘΘω
*model/multi_head_attention/einsum_1/EinsumEinsum6model/multi_head_attention/dropout_9/Identity:output:0(model/multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationacbe,aecd->abcdή
Hmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpQmodel_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0§
9model/multi_head_attention/attention_output/einsum/EinsumEinsum3model/multi_head_attention/einsum_1/Einsum:output:0Pmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????Θ*
equationabcd,cde->abeΒ
>model/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpGmodel_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0ϋ
/model/multi_head_attention/attention_output/addAddV2Bmodel/multi_head_attention/attention_output/einsum/Einsum:output:0Fmodel/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ
model/dropout/IdentityIdentity3model/multi_head_attention/attention_output/add:z:0*
T0*,
_output_shapes
:?????????Θ
 model/tf.__operators__.add/AddV2AddV2model/dropout/Identity:output:0input_1*
T0*,
_output_shapes
:?????????Θ
:model/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:γ
(model/layer_normalization_1/moments/meanMean$model/tf.__operators__.add/AddV2:z:0Cmodel/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(ͺ
0model/layer_normalization_1/moments/StopGradientStopGradient1model/layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:?????????Θβ
5model/layer_normalization_1/moments/SquaredDifferenceSquaredDifference$model/tf.__operators__.add/AddV2:z:09model/layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θ
>model/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
,model/layer_normalization_1/moments/varianceMean9model/layer_normalization_1/moments/SquaredDifference:z:0Gmodel/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(p
+model/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Φ
)model/layer_normalization_1/batchnorm/addAddV25model/layer_normalization_1/moments/variance:output:04model/layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θ
+model/layer_normalization_1/batchnorm/RsqrtRsqrt-model/layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:?????????ΘΆ
8model/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Ϊ
)model/layer_normalization_1/batchnorm/mulMul/model/layer_normalization_1/batchnorm/Rsqrt:y:0@model/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????ΘΎ
+model/layer_normalization_1/batchnorm/mul_1Mul$model/tf.__operators__.add/AddV2:z:0-model/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ΘΛ
+model/layer_normalization_1/batchnorm/mul_2Mul1model/layer_normalization_1/moments/mean:output:0-model/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θ?
4model/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp=model_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Φ
)model/layer_normalization_1/batchnorm/subSub<model/layer_normalization_1/batchnorm/ReadVariableOp:value:0/model/layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????ΘΛ
+model/layer_normalization_1/batchnorm/add_1AddV2/model/layer_normalization_1/batchnorm/mul_1:z:0-model/layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Θm
"model/conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????Ε
model/conv1d/Conv1D/ExpandDims
ExpandDims/model/layer_normalization_1/batchnorm/add_1:z:0+model/conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ¬
/model/conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp8model_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0f
$model/conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Η
 model/conv1d/Conv1D/ExpandDims_1
ExpandDims7model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0-model/conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Υ
model/conv1d/Conv1DConv2D'model/conv1d/Conv1D/ExpandDims:output:0)model/conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

model/conv1d/Conv1D/SqueezeSqueezemodel/conv1d/Conv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????
#model/conv1d/BiasAdd/ReadVariableOpReadVariableOp,model_conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0©
model/conv1d/BiasAddBiasAdd$model/conv1d/Conv1D/Squeeze:output:0+model/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θo
model/conv1d/ReluRelumodel/conv1d/BiasAdd:output:0*
T0*,
_output_shapes
:?????????Θ|
model/dropout_1/IdentityIdentitymodel/conv1d/Relu:activations:0*
T0*,
_output_shapes
:?????????Θo
$model/conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????»
 model/conv1d_1/Conv1D/ExpandDims
ExpandDims!model/dropout_1/Identity:output:0-model/conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ°
1model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0h
&model/conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Ν
"model/conv1d_1/Conv1D/ExpandDims_1
ExpandDims9model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Ϋ
model/conv1d_1/Conv1DConv2D)model/conv1d_1/Conv1D/ExpandDims:output:0+model/conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

model/conv1d_1/Conv1D/SqueezeSqueezemodel/conv1d_1/Conv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????
%model/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0―
model/conv1d_1/BiasAddBiasAdd&model/conv1d_1/Conv1D/Squeeze:output:0-model/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ©
"model/tf.__operators__.add_1/AddV2AddV2model/conv1d_1/BiasAdd:output:0$model/tf.__operators__.add/AddV2:z:0*
T0*,
_output_shapes
:?????????Θ
:model/layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ε
(model/layer_normalization_2/moments/meanMean&model/tf.__operators__.add_1/AddV2:z:0Cmodel/layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(ͺ
0model/layer_normalization_2/moments/StopGradientStopGradient1model/layer_normalization_2/moments/mean:output:0*
T0*,
_output_shapes
:?????????Θδ
5model/layer_normalization_2/moments/SquaredDifferenceSquaredDifference&model/tf.__operators__.add_1/AddV2:z:09model/layer_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θ
>model/layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
,model/layer_normalization_2/moments/varianceMean9model/layer_normalization_2/moments/SquaredDifference:z:0Gmodel/layer_normalization_2/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(p
+model/layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Φ
)model/layer_normalization_2/batchnorm/addAddV25model/layer_normalization_2/moments/variance:output:04model/layer_normalization_2/batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θ
+model/layer_normalization_2/batchnorm/RsqrtRsqrt-model/layer_normalization_2/batchnorm/add:z:0*
T0*,
_output_shapes
:?????????ΘΆ
8model/layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Ϊ
)model/layer_normalization_2/batchnorm/mulMul/model/layer_normalization_2/batchnorm/Rsqrt:y:0@model/layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θΐ
+model/layer_normalization_2/batchnorm/mul_1Mul&model/tf.__operators__.add_1/AddV2:z:0-model/layer_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ΘΛ
+model/layer_normalization_2/batchnorm/mul_2Mul1model/layer_normalization_2/moments/mean:output:0-model/layer_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θ?
4model/layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp=model_layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Φ
)model/layer_normalization_2/batchnorm/subSub<model/layer_normalization_2/batchnorm/ReadVariableOp:value:0/model/layer_normalization_2/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????ΘΛ
+model/layer_normalization_2/batchnorm/add_1AddV2/model/layer_normalization_2/batchnorm/mul_1:z:0-model/layer_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????ΘΜ
?model/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOpHmodel_multi_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0
0model/multi_head_attention_1/query/einsum/EinsumEinsum/model/layer_normalization_2/batchnorm/add_1:z:0Gmodel/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde΄
5model/multi_head_attention_1/query/add/ReadVariableOpReadVariableOp>model_multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

:2*
dtype0δ
&model/multi_head_attention_1/query/addAddV29model/multi_head_attention_1/query/einsum/Einsum:output:0=model/multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2Θ
=model/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOpFmodel_multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0
.model/multi_head_attention_1/key/einsum/EinsumEinsum/model/layer_normalization_2/batchnorm/add_1:z:0Emodel/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde°
3model/multi_head_attention_1/key/add/ReadVariableOpReadVariableOp<model_multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

:2*
dtype0ή
$model/multi_head_attention_1/key/addAddV27model/multi_head_attention_1/key/einsum/Einsum:output:0;model/multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2Μ
?model/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOpHmodel_multi_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0
0model/multi_head_attention_1/value/einsum/EinsumEinsum/model/layer_normalization_2/batchnorm/add_1:z:0Gmodel/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde΄
5model/multi_head_attention_1/value/add/ReadVariableOpReadVariableOp>model_multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

:2*
dtype0δ
&model/multi_head_attention_1/value/addAddV29model/multi_head_attention_1/value/einsum/Einsum:output:0=model/multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2g
"model/multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΓΠ>»
 model/multi_head_attention_1/MulMul*model/multi_head_attention_1/query/add:z:0+model/multi_head_attention_1/Mul/y:output:0*
T0*0
_output_shapes
:?????????Θ2θ
*model/multi_head_attention_1/einsum/EinsumEinsum(model/multi_head_attention_1/key/add:z:0$model/multi_head_attention_1/Mul:z:0*
N*
T0*1
_output_shapes
:?????????ΘΘ*
equationaecd,abcd->acbeͺ
.model/multi_head_attention_1/softmax_1/SoftmaxSoftmax3model/multi_head_attention_1/einsum/Einsum:output:0*
T0*1
_output_shapes
:?????????ΘΘ²
0model/multi_head_attention_1/dropout_10/IdentityIdentity8model/multi_head_attention_1/softmax_1/Softmax:softmax:0*
T0*1
_output_shapes
:?????????ΘΘ
,model/multi_head_attention_1/einsum_1/EinsumEinsum9model/multi_head_attention_1/dropout_10/Identity:output:0*model/multi_head_attention_1/value/add:z:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationacbe,aecd->abcdβ
Jmodel/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpSmodel_multi_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0­
;model/multi_head_attention_1/attention_output/einsum/EinsumEinsum5model/multi_head_attention_1/einsum_1/Einsum:output:0Rmodel/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????Θ*
equationabcd,cde->abeΖ
@model/multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOpImodel_multi_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0
1model/multi_head_attention_1/attention_output/addAddV2Dmodel/multi_head_attention_1/attention_output/einsum/Einsum:output:0Hmodel/multi_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ
model/dropout_2/IdentityIdentity5model/multi_head_attention_1/attention_output/add:z:0*
T0*,
_output_shapes
:?????????Θ­
"model/tf.__operators__.add_2/AddV2AddV2!model/dropout_2/Identity:output:0&model/tf.__operators__.add_1/AddV2:z:0*
T0*,
_output_shapes
:?????????Θ
:model/layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ε
(model/layer_normalization_3/moments/meanMean&model/tf.__operators__.add_2/AddV2:z:0Cmodel/layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(ͺ
0model/layer_normalization_3/moments/StopGradientStopGradient1model/layer_normalization_3/moments/mean:output:0*
T0*,
_output_shapes
:?????????Θδ
5model/layer_normalization_3/moments/SquaredDifferenceSquaredDifference&model/tf.__operators__.add_2/AddV2:z:09model/layer_normalization_3/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θ
>model/layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
,model/layer_normalization_3/moments/varianceMean9model/layer_normalization_3/moments/SquaredDifference:z:0Gmodel/layer_normalization_3/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(p
+model/layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Φ
)model/layer_normalization_3/batchnorm/addAddV25model/layer_normalization_3/moments/variance:output:04model/layer_normalization_3/batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θ
+model/layer_normalization_3/batchnorm/RsqrtRsqrt-model/layer_normalization_3/batchnorm/add:z:0*
T0*,
_output_shapes
:?????????ΘΆ
8model/layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Ϊ
)model/layer_normalization_3/batchnorm/mulMul/model/layer_normalization_3/batchnorm/Rsqrt:y:0@model/layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θΐ
+model/layer_normalization_3/batchnorm/mul_1Mul&model/tf.__operators__.add_2/AddV2:z:0-model/layer_normalization_3/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ΘΛ
+model/layer_normalization_3/batchnorm/mul_2Mul1model/layer_normalization_3/moments/mean:output:0-model/layer_normalization_3/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θ?
4model/layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp=model_layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Φ
)model/layer_normalization_3/batchnorm/subSub<model/layer_normalization_3/batchnorm/ReadVariableOp:value:0/model/layer_normalization_3/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????ΘΛ
+model/layer_normalization_3/batchnorm/add_1AddV2/model/layer_normalization_3/batchnorm/mul_1:z:0-model/layer_normalization_3/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Θo
$model/conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????Ι
 model/conv1d_2/Conv1D/ExpandDims
ExpandDims/model/layer_normalization_3/batchnorm/add_1:z:0-model/conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ°
1model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0h
&model/conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Ν
"model/conv1d_2/Conv1D/ExpandDims_1
ExpandDims9model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Ϋ
model/conv1d_2/Conv1DConv2D)model/conv1d_2/Conv1D/ExpandDims:output:0+model/conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

model/conv1d_2/Conv1D/SqueezeSqueezemodel/conv1d_2/Conv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????
%model/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0―
model/conv1d_2/BiasAddBiasAdd&model/conv1d_2/Conv1D/Squeeze:output:0-model/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θs
model/conv1d_2/ReluRelumodel/conv1d_2/BiasAdd:output:0*
T0*,
_output_shapes
:?????????Θ~
model/dropout_3/IdentityIdentity!model/conv1d_2/Relu:activations:0*
T0*,
_output_shapes
:?????????Θo
$model/conv1d_3/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????»
 model/conv1d_3/Conv1D/ExpandDims
ExpandDims!model/dropout_3/Identity:output:0-model/conv1d_3/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ°
1model/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0h
&model/conv1d_3/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Ν
"model/conv1d_3/Conv1D/ExpandDims_1
ExpandDims9model/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_3/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Ϋ
model/conv1d_3/Conv1DConv2D)model/conv1d_3/Conv1D/ExpandDims:output:0+model/conv1d_3/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

model/conv1d_3/Conv1D/SqueezeSqueezemodel/conv1d_3/Conv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????
%model/conv1d_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0―
model/conv1d_3/BiasAddBiasAdd&model/conv1d_3/Conv1D/Squeeze:output:0-model/conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ«
"model/tf.__operators__.add_3/AddV2AddV2model/conv1d_3/BiasAdd:output:0&model/tf.__operators__.add_2/AddV2:z:0*
T0*,
_output_shapes
:?????????Θ
:model/layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ε
(model/layer_normalization_4/moments/meanMean&model/tf.__operators__.add_3/AddV2:z:0Cmodel/layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(ͺ
0model/layer_normalization_4/moments/StopGradientStopGradient1model/layer_normalization_4/moments/mean:output:0*
T0*,
_output_shapes
:?????????Θδ
5model/layer_normalization_4/moments/SquaredDifferenceSquaredDifference&model/tf.__operators__.add_3/AddV2:z:09model/layer_normalization_4/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θ
>model/layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
,model/layer_normalization_4/moments/varianceMean9model/layer_normalization_4/moments/SquaredDifference:z:0Gmodel/layer_normalization_4/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(p
+model/layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Φ
)model/layer_normalization_4/batchnorm/addAddV25model/layer_normalization_4/moments/variance:output:04model/layer_normalization_4/batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θ
+model/layer_normalization_4/batchnorm/RsqrtRsqrt-model/layer_normalization_4/batchnorm/add:z:0*
T0*,
_output_shapes
:?????????ΘΆ
8model/layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Ϊ
)model/layer_normalization_4/batchnorm/mulMul/model/layer_normalization_4/batchnorm/Rsqrt:y:0@model/layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θΐ
+model/layer_normalization_4/batchnorm/mul_1Mul&model/tf.__operators__.add_3/AddV2:z:0-model/layer_normalization_4/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ΘΛ
+model/layer_normalization_4/batchnorm/mul_2Mul1model/layer_normalization_4/moments/mean:output:0-model/layer_normalization_4/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θ?
4model/layer_normalization_4/batchnorm/ReadVariableOpReadVariableOp=model_layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Φ
)model/layer_normalization_4/batchnorm/subSub<model/layer_normalization_4/batchnorm/ReadVariableOp:value:0/model/layer_normalization_4/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????ΘΛ
+model/layer_normalization_4/batchnorm/add_1AddV2/model/layer_normalization_4/batchnorm/mul_1:z:0-model/layer_normalization_4/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????ΘΜ
?model/multi_head_attention_2/query/einsum/Einsum/ReadVariableOpReadVariableOpHmodel_multi_head_attention_2_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0
0model/multi_head_attention_2/query/einsum/EinsumEinsum/model/layer_normalization_4/batchnorm/add_1:z:0Gmodel/multi_head_attention_2/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde΄
5model/multi_head_attention_2/query/add/ReadVariableOpReadVariableOp>model_multi_head_attention_2_query_add_readvariableop_resource*
_output_shapes

:2*
dtype0δ
&model/multi_head_attention_2/query/addAddV29model/multi_head_attention_2/query/einsum/Einsum:output:0=model/multi_head_attention_2/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2Θ
=model/multi_head_attention_2/key/einsum/Einsum/ReadVariableOpReadVariableOpFmodel_multi_head_attention_2_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0
.model/multi_head_attention_2/key/einsum/EinsumEinsum/model/layer_normalization_4/batchnorm/add_1:z:0Emodel/multi_head_attention_2/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde°
3model/multi_head_attention_2/key/add/ReadVariableOpReadVariableOp<model_multi_head_attention_2_key_add_readvariableop_resource*
_output_shapes

:2*
dtype0ή
$model/multi_head_attention_2/key/addAddV27model/multi_head_attention_2/key/einsum/Einsum:output:0;model/multi_head_attention_2/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2Μ
?model/multi_head_attention_2/value/einsum/Einsum/ReadVariableOpReadVariableOpHmodel_multi_head_attention_2_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0
0model/multi_head_attention_2/value/einsum/EinsumEinsum/model/layer_normalization_4/batchnorm/add_1:z:0Gmodel/multi_head_attention_2/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde΄
5model/multi_head_attention_2/value/add/ReadVariableOpReadVariableOp>model_multi_head_attention_2_value_add_readvariableop_resource*
_output_shapes

:2*
dtype0δ
&model/multi_head_attention_2/value/addAddV29model/multi_head_attention_2/value/einsum/Einsum:output:0=model/multi_head_attention_2/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2g
"model/multi_head_attention_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΓΠ>»
 model/multi_head_attention_2/MulMul*model/multi_head_attention_2/query/add:z:0+model/multi_head_attention_2/Mul/y:output:0*
T0*0
_output_shapes
:?????????Θ2θ
*model/multi_head_attention_2/einsum/EinsumEinsum(model/multi_head_attention_2/key/add:z:0$model/multi_head_attention_2/Mul:z:0*
N*
T0*1
_output_shapes
:?????????ΘΘ*
equationaecd,abcd->acbeͺ
.model/multi_head_attention_2/softmax_2/SoftmaxSoftmax3model/multi_head_attention_2/einsum/Einsum:output:0*
T0*1
_output_shapes
:?????????ΘΘ²
0model/multi_head_attention_2/dropout_11/IdentityIdentity8model/multi_head_attention_2/softmax_2/Softmax:softmax:0*
T0*1
_output_shapes
:?????????ΘΘ
,model/multi_head_attention_2/einsum_1/EinsumEinsum9model/multi_head_attention_2/dropout_11/Identity:output:0*model/multi_head_attention_2/value/add:z:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationacbe,aecd->abcdβ
Jmodel/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpSmodel_multi_head_attention_2_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0­
;model/multi_head_attention_2/attention_output/einsum/EinsumEinsum5model/multi_head_attention_2/einsum_1/Einsum:output:0Rmodel/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????Θ*
equationabcd,cde->abeΖ
@model/multi_head_attention_2/attention_output/add/ReadVariableOpReadVariableOpImodel_multi_head_attention_2_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0
1model/multi_head_attention_2/attention_output/addAddV2Dmodel/multi_head_attention_2/attention_output/einsum/Einsum:output:0Hmodel/multi_head_attention_2/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ
model/dropout_4/IdentityIdentity5model/multi_head_attention_2/attention_output/add:z:0*
T0*,
_output_shapes
:?????????Θ­
"model/tf.__operators__.add_4/AddV2AddV2!model/dropout_4/Identity:output:0&model/tf.__operators__.add_3/AddV2:z:0*
T0*,
_output_shapes
:?????????Θ
:model/layer_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ε
(model/layer_normalization_5/moments/meanMean&model/tf.__operators__.add_4/AddV2:z:0Cmodel/layer_normalization_5/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(ͺ
0model/layer_normalization_5/moments/StopGradientStopGradient1model/layer_normalization_5/moments/mean:output:0*
T0*,
_output_shapes
:?????????Θδ
5model/layer_normalization_5/moments/SquaredDifferenceSquaredDifference&model/tf.__operators__.add_4/AddV2:z:09model/layer_normalization_5/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θ
>model/layer_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
,model/layer_normalization_5/moments/varianceMean9model/layer_normalization_5/moments/SquaredDifference:z:0Gmodel/layer_normalization_5/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(p
+model/layer_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Φ
)model/layer_normalization_5/batchnorm/addAddV25model/layer_normalization_5/moments/variance:output:04model/layer_normalization_5/batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θ
+model/layer_normalization_5/batchnorm/RsqrtRsqrt-model/layer_normalization_5/batchnorm/add:z:0*
T0*,
_output_shapes
:?????????ΘΆ
8model/layer_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_layer_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Ϊ
)model/layer_normalization_5/batchnorm/mulMul/model/layer_normalization_5/batchnorm/Rsqrt:y:0@model/layer_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θΐ
+model/layer_normalization_5/batchnorm/mul_1Mul&model/tf.__operators__.add_4/AddV2:z:0-model/layer_normalization_5/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ΘΛ
+model/layer_normalization_5/batchnorm/mul_2Mul1model/layer_normalization_5/moments/mean:output:0-model/layer_normalization_5/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θ?
4model/layer_normalization_5/batchnorm/ReadVariableOpReadVariableOp=model_layer_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Φ
)model/layer_normalization_5/batchnorm/subSub<model/layer_normalization_5/batchnorm/ReadVariableOp:value:0/model/layer_normalization_5/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????ΘΛ
+model/layer_normalization_5/batchnorm/add_1AddV2/model/layer_normalization_5/batchnorm/mul_1:z:0-model/layer_normalization_5/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Θo
$model/conv1d_4/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????Ι
 model/conv1d_4/Conv1D/ExpandDims
ExpandDims/model/layer_normalization_5/batchnorm/add_1:z:0-model/conv1d_4/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ°
1model/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0h
&model/conv1d_4/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Ν
"model/conv1d_4/Conv1D/ExpandDims_1
ExpandDims9model/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_4/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Ϋ
model/conv1d_4/Conv1DConv2D)model/conv1d_4/Conv1D/ExpandDims:output:0+model/conv1d_4/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

model/conv1d_4/Conv1D/SqueezeSqueezemodel/conv1d_4/Conv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????
%model/conv1d_4/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0―
model/conv1d_4/BiasAddBiasAdd&model/conv1d_4/Conv1D/Squeeze:output:0-model/conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θs
model/conv1d_4/ReluRelumodel/conv1d_4/BiasAdd:output:0*
T0*,
_output_shapes
:?????????Θ~
model/dropout_5/IdentityIdentity!model/conv1d_4/Relu:activations:0*
T0*,
_output_shapes
:?????????Θo
$model/conv1d_5/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????»
 model/conv1d_5/Conv1D/ExpandDims
ExpandDims!model/dropout_5/Identity:output:0-model/conv1d_5/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ°
1model/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0h
&model/conv1d_5/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Ν
"model/conv1d_5/Conv1D/ExpandDims_1
ExpandDims9model/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_5/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Ϋ
model/conv1d_5/Conv1DConv2D)model/conv1d_5/Conv1D/ExpandDims:output:0+model/conv1d_5/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

model/conv1d_5/Conv1D/SqueezeSqueezemodel/conv1d_5/Conv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????
%model/conv1d_5/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0―
model/conv1d_5/BiasAddBiasAdd&model/conv1d_5/Conv1D/Squeeze:output:0-model/conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ«
"model/tf.__operators__.add_5/AddV2AddV2model/conv1d_5/BiasAdd:output:0&model/tf.__operators__.add_4/AddV2:z:0*
T0*,
_output_shapes
:?????????Θ
:model/layer_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ε
(model/layer_normalization_6/moments/meanMean&model/tf.__operators__.add_5/AddV2:z:0Cmodel/layer_normalization_6/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(ͺ
0model/layer_normalization_6/moments/StopGradientStopGradient1model/layer_normalization_6/moments/mean:output:0*
T0*,
_output_shapes
:?????????Θδ
5model/layer_normalization_6/moments/SquaredDifferenceSquaredDifference&model/tf.__operators__.add_5/AddV2:z:09model/layer_normalization_6/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θ
>model/layer_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
,model/layer_normalization_6/moments/varianceMean9model/layer_normalization_6/moments/SquaredDifference:z:0Gmodel/layer_normalization_6/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(p
+model/layer_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Φ
)model/layer_normalization_6/batchnorm/addAddV25model/layer_normalization_6/moments/variance:output:04model/layer_normalization_6/batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θ
+model/layer_normalization_6/batchnorm/RsqrtRsqrt-model/layer_normalization_6/batchnorm/add:z:0*
T0*,
_output_shapes
:?????????ΘΆ
8model/layer_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_layer_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Ϊ
)model/layer_normalization_6/batchnorm/mulMul/model/layer_normalization_6/batchnorm/Rsqrt:y:0@model/layer_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θΐ
+model/layer_normalization_6/batchnorm/mul_1Mul&model/tf.__operators__.add_5/AddV2:z:0-model/layer_normalization_6/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ΘΛ
+model/layer_normalization_6/batchnorm/mul_2Mul1model/layer_normalization_6/moments/mean:output:0-model/layer_normalization_6/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θ?
4model/layer_normalization_6/batchnorm/ReadVariableOpReadVariableOp=model_layer_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Φ
)model/layer_normalization_6/batchnorm/subSub<model/layer_normalization_6/batchnorm/ReadVariableOp:value:0/model/layer_normalization_6/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????ΘΛ
+model/layer_normalization_6/batchnorm/add_1AddV2/model/layer_normalization_6/batchnorm/mul_1:z:0-model/layer_normalization_6/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????ΘΜ
?model/multi_head_attention_3/query/einsum/Einsum/ReadVariableOpReadVariableOpHmodel_multi_head_attention_3_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0
0model/multi_head_attention_3/query/einsum/EinsumEinsum/model/layer_normalization_6/batchnorm/add_1:z:0Gmodel/multi_head_attention_3/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde΄
5model/multi_head_attention_3/query/add/ReadVariableOpReadVariableOp>model_multi_head_attention_3_query_add_readvariableop_resource*
_output_shapes

:2*
dtype0δ
&model/multi_head_attention_3/query/addAddV29model/multi_head_attention_3/query/einsum/Einsum:output:0=model/multi_head_attention_3/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2Θ
=model/multi_head_attention_3/key/einsum/Einsum/ReadVariableOpReadVariableOpFmodel_multi_head_attention_3_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0
.model/multi_head_attention_3/key/einsum/EinsumEinsum/model/layer_normalization_6/batchnorm/add_1:z:0Emodel/multi_head_attention_3/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde°
3model/multi_head_attention_3/key/add/ReadVariableOpReadVariableOp<model_multi_head_attention_3_key_add_readvariableop_resource*
_output_shapes

:2*
dtype0ή
$model/multi_head_attention_3/key/addAddV27model/multi_head_attention_3/key/einsum/Einsum:output:0;model/multi_head_attention_3/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2Μ
?model/multi_head_attention_3/value/einsum/Einsum/ReadVariableOpReadVariableOpHmodel_multi_head_attention_3_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0
0model/multi_head_attention_3/value/einsum/EinsumEinsum/model/layer_normalization_6/batchnorm/add_1:z:0Gmodel/multi_head_attention_3/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde΄
5model/multi_head_attention_3/value/add/ReadVariableOpReadVariableOp>model_multi_head_attention_3_value_add_readvariableop_resource*
_output_shapes

:2*
dtype0δ
&model/multi_head_attention_3/value/addAddV29model/multi_head_attention_3/value/einsum/Einsum:output:0=model/multi_head_attention_3/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2g
"model/multi_head_attention_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΓΠ>»
 model/multi_head_attention_3/MulMul*model/multi_head_attention_3/query/add:z:0+model/multi_head_attention_3/Mul/y:output:0*
T0*0
_output_shapes
:?????????Θ2θ
*model/multi_head_attention_3/einsum/EinsumEinsum(model/multi_head_attention_3/key/add:z:0$model/multi_head_attention_3/Mul:z:0*
N*
T0*1
_output_shapes
:?????????ΘΘ*
equationaecd,abcd->acbeͺ
.model/multi_head_attention_3/softmax_3/SoftmaxSoftmax3model/multi_head_attention_3/einsum/Einsum:output:0*
T0*1
_output_shapes
:?????????ΘΘ²
0model/multi_head_attention_3/dropout_12/IdentityIdentity8model/multi_head_attention_3/softmax_3/Softmax:softmax:0*
T0*1
_output_shapes
:?????????ΘΘ
,model/multi_head_attention_3/einsum_1/EinsumEinsum9model/multi_head_attention_3/dropout_12/Identity:output:0*model/multi_head_attention_3/value/add:z:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationacbe,aecd->abcdβ
Jmodel/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpSmodel_multi_head_attention_3_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0­
;model/multi_head_attention_3/attention_output/einsum/EinsumEinsum5model/multi_head_attention_3/einsum_1/Einsum:output:0Rmodel/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????Θ*
equationabcd,cde->abeΖ
@model/multi_head_attention_3/attention_output/add/ReadVariableOpReadVariableOpImodel_multi_head_attention_3_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0
1model/multi_head_attention_3/attention_output/addAddV2Dmodel/multi_head_attention_3/attention_output/einsum/Einsum:output:0Hmodel/multi_head_attention_3/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ
model/dropout_6/IdentityIdentity5model/multi_head_attention_3/attention_output/add:z:0*
T0*,
_output_shapes
:?????????Θ­
"model/tf.__operators__.add_6/AddV2AddV2!model/dropout_6/Identity:output:0&model/tf.__operators__.add_5/AddV2:z:0*
T0*,
_output_shapes
:?????????Θ
:model/layer_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ε
(model/layer_normalization_7/moments/meanMean&model/tf.__operators__.add_6/AddV2:z:0Cmodel/layer_normalization_7/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(ͺ
0model/layer_normalization_7/moments/StopGradientStopGradient1model/layer_normalization_7/moments/mean:output:0*
T0*,
_output_shapes
:?????????Θδ
5model/layer_normalization_7/moments/SquaredDifferenceSquaredDifference&model/tf.__operators__.add_6/AddV2:z:09model/layer_normalization_7/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θ
>model/layer_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
,model/layer_normalization_7/moments/varianceMean9model/layer_normalization_7/moments/SquaredDifference:z:0Gmodel/layer_normalization_7/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(p
+model/layer_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Φ
)model/layer_normalization_7/batchnorm/addAddV25model/layer_normalization_7/moments/variance:output:04model/layer_normalization_7/batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θ
+model/layer_normalization_7/batchnorm/RsqrtRsqrt-model/layer_normalization_7/batchnorm/add:z:0*
T0*,
_output_shapes
:?????????ΘΆ
8model/layer_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_layer_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Ϊ
)model/layer_normalization_7/batchnorm/mulMul/model/layer_normalization_7/batchnorm/Rsqrt:y:0@model/layer_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θΐ
+model/layer_normalization_7/batchnorm/mul_1Mul&model/tf.__operators__.add_6/AddV2:z:0-model/layer_normalization_7/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ΘΛ
+model/layer_normalization_7/batchnorm/mul_2Mul1model/layer_normalization_7/moments/mean:output:0-model/layer_normalization_7/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θ?
4model/layer_normalization_7/batchnorm/ReadVariableOpReadVariableOp=model_layer_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Φ
)model/layer_normalization_7/batchnorm/subSub<model/layer_normalization_7/batchnorm/ReadVariableOp:value:0/model/layer_normalization_7/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????ΘΛ
+model/layer_normalization_7/batchnorm/add_1AddV2/model/layer_normalization_7/batchnorm/mul_1:z:0-model/layer_normalization_7/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Θo
$model/conv1d_6/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????Ι
 model/conv1d_6/Conv1D/ExpandDims
ExpandDims/model/layer_normalization_7/batchnorm/add_1:z:0-model/conv1d_6/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ°
1model/conv1d_6/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0h
&model/conv1d_6/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Ν
"model/conv1d_6/Conv1D/ExpandDims_1
ExpandDims9model/conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_6/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Ϋ
model/conv1d_6/Conv1DConv2D)model/conv1d_6/Conv1D/ExpandDims:output:0+model/conv1d_6/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

model/conv1d_6/Conv1D/SqueezeSqueezemodel/conv1d_6/Conv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????
%model/conv1d_6/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0―
model/conv1d_6/BiasAddBiasAdd&model/conv1d_6/Conv1D/Squeeze:output:0-model/conv1d_6/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θs
model/conv1d_6/ReluRelumodel/conv1d_6/BiasAdd:output:0*
T0*,
_output_shapes
:?????????Θ~
model/dropout_7/IdentityIdentity!model/conv1d_6/Relu:activations:0*
T0*,
_output_shapes
:?????????Θo
$model/conv1d_7/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????»
 model/conv1d_7/Conv1D/ExpandDims
ExpandDims!model/dropout_7/Identity:output:0-model/conv1d_7/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ°
1model/conv1d_7/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_7_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0h
&model/conv1d_7/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Ν
"model/conv1d_7/Conv1D/ExpandDims_1
ExpandDims9model/conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_7/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Ϋ
model/conv1d_7/Conv1DConv2D)model/conv1d_7/Conv1D/ExpandDims:output:0+model/conv1d_7/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

model/conv1d_7/Conv1D/SqueezeSqueezemodel/conv1d_7/Conv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????
%model/conv1d_7/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0―
model/conv1d_7/BiasAddBiasAdd&model/conv1d_7/Conv1D/Squeeze:output:0-model/conv1d_7/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ«
"model/tf.__operators__.add_7/AddV2AddV2model/conv1d_7/BiasAdd:output:0&model/tf.__operators__.add_6/AddV2:z:0*
T0*,
_output_shapes
:?????????Θw
5model/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Ζ
#model/global_average_pooling1d/MeanMean&model/tf.__operators__.add_7/AddV2:z:0>model/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:?????????Θ
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource* 
_output_shapes
:
Θ*
dtype0¨
model/dense/MatMulMatMul,model/global_average_pooling1d/Mean:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????i
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????w
model/dropout_8/IdentityIdentitymodel/dense/Relu:activations:0*
T0*(
_output_shapes
:?????????
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0 
model/dense_1/MatMulMatMul!model/dropout_8/Identity:output:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0 
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m
IdentityIdentitymodel/dense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
NoOpNoOp$^model/conv1d/BiasAdd/ReadVariableOp0^model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp&^model/conv1d_1/BiasAdd/ReadVariableOp2^model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp&^model/conv1d_2/BiasAdd/ReadVariableOp2^model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp&^model/conv1d_3/BiasAdd/ReadVariableOp2^model/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp&^model/conv1d_4/BiasAdd/ReadVariableOp2^model/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp&^model/conv1d_5/BiasAdd/ReadVariableOp2^model/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp&^model/conv1d_6/BiasAdd/ReadVariableOp2^model/conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp&^model/conv1d_7/BiasAdd/ReadVariableOp2^model/conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp3^model/layer_normalization/batchnorm/ReadVariableOp7^model/layer_normalization/batchnorm/mul/ReadVariableOp5^model/layer_normalization_1/batchnorm/ReadVariableOp9^model/layer_normalization_1/batchnorm/mul/ReadVariableOp5^model/layer_normalization_2/batchnorm/ReadVariableOp9^model/layer_normalization_2/batchnorm/mul/ReadVariableOp5^model/layer_normalization_3/batchnorm/ReadVariableOp9^model/layer_normalization_3/batchnorm/mul/ReadVariableOp5^model/layer_normalization_4/batchnorm/ReadVariableOp9^model/layer_normalization_4/batchnorm/mul/ReadVariableOp5^model/layer_normalization_5/batchnorm/ReadVariableOp9^model/layer_normalization_5/batchnorm/mul/ReadVariableOp5^model/layer_normalization_6/batchnorm/ReadVariableOp9^model/layer_normalization_6/batchnorm/mul/ReadVariableOp5^model/layer_normalization_7/batchnorm/ReadVariableOp9^model/layer_normalization_7/batchnorm/mul/ReadVariableOp?^model/multi_head_attention/attention_output/add/ReadVariableOpI^model/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2^model/multi_head_attention/key/add/ReadVariableOp<^model/multi_head_attention/key/einsum/Einsum/ReadVariableOp4^model/multi_head_attention/query/add/ReadVariableOp>^model/multi_head_attention/query/einsum/Einsum/ReadVariableOp4^model/multi_head_attention/value/add/ReadVariableOp>^model/multi_head_attention/value/einsum/Einsum/ReadVariableOpA^model/multi_head_attention_1/attention_output/add/ReadVariableOpK^model/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp4^model/multi_head_attention_1/key/add/ReadVariableOp>^model/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp6^model/multi_head_attention_1/query/add/ReadVariableOp@^model/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp6^model/multi_head_attention_1/value/add/ReadVariableOp@^model/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpA^model/multi_head_attention_2/attention_output/add/ReadVariableOpK^model/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp4^model/multi_head_attention_2/key/add/ReadVariableOp>^model/multi_head_attention_2/key/einsum/Einsum/ReadVariableOp6^model/multi_head_attention_2/query/add/ReadVariableOp@^model/multi_head_attention_2/query/einsum/Einsum/ReadVariableOp6^model/multi_head_attention_2/value/add/ReadVariableOp@^model/multi_head_attention_2/value/einsum/Einsum/ReadVariableOpA^model/multi_head_attention_3/attention_output/add/ReadVariableOpK^model/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp4^model/multi_head_attention_3/key/add/ReadVariableOp>^model/multi_head_attention_3/key/einsum/Einsum/ReadVariableOp6^model/multi_head_attention_3/query/add/ReadVariableOp@^model/multi_head_attention_3/query/einsum/Einsum/ReadVariableOp6^model/multi_head_attention_3/value/add/ReadVariableOp@^model/multi_head_attention_3/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*΅
_input_shapes£
 :?????????Θ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#model/conv1d/BiasAdd/ReadVariableOp#model/conv1d/BiasAdd/ReadVariableOp2b
/model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp/model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp2N
%model/conv1d_1/BiasAdd/ReadVariableOp%model/conv1d_1/BiasAdd/ReadVariableOp2f
1model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp1model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2N
%model/conv1d_2/BiasAdd/ReadVariableOp%model/conv1d_2/BiasAdd/ReadVariableOp2f
1model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp1model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2N
%model/conv1d_3/BiasAdd/ReadVariableOp%model/conv1d_3/BiasAdd/ReadVariableOp2f
1model/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp1model/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp2N
%model/conv1d_4/BiasAdd/ReadVariableOp%model/conv1d_4/BiasAdd/ReadVariableOp2f
1model/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp1model/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp2N
%model/conv1d_5/BiasAdd/ReadVariableOp%model/conv1d_5/BiasAdd/ReadVariableOp2f
1model/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp1model/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp2N
%model/conv1d_6/BiasAdd/ReadVariableOp%model/conv1d_6/BiasAdd/ReadVariableOp2f
1model/conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp1model/conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp2N
%model/conv1d_7/BiasAdd/ReadVariableOp%model/conv1d_7/BiasAdd/ReadVariableOp2f
1model/conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp1model/conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2h
2model/layer_normalization/batchnorm/ReadVariableOp2model/layer_normalization/batchnorm/ReadVariableOp2p
6model/layer_normalization/batchnorm/mul/ReadVariableOp6model/layer_normalization/batchnorm/mul/ReadVariableOp2l
4model/layer_normalization_1/batchnorm/ReadVariableOp4model/layer_normalization_1/batchnorm/ReadVariableOp2t
8model/layer_normalization_1/batchnorm/mul/ReadVariableOp8model/layer_normalization_1/batchnorm/mul/ReadVariableOp2l
4model/layer_normalization_2/batchnorm/ReadVariableOp4model/layer_normalization_2/batchnorm/ReadVariableOp2t
8model/layer_normalization_2/batchnorm/mul/ReadVariableOp8model/layer_normalization_2/batchnorm/mul/ReadVariableOp2l
4model/layer_normalization_3/batchnorm/ReadVariableOp4model/layer_normalization_3/batchnorm/ReadVariableOp2t
8model/layer_normalization_3/batchnorm/mul/ReadVariableOp8model/layer_normalization_3/batchnorm/mul/ReadVariableOp2l
4model/layer_normalization_4/batchnorm/ReadVariableOp4model/layer_normalization_4/batchnorm/ReadVariableOp2t
8model/layer_normalization_4/batchnorm/mul/ReadVariableOp8model/layer_normalization_4/batchnorm/mul/ReadVariableOp2l
4model/layer_normalization_5/batchnorm/ReadVariableOp4model/layer_normalization_5/batchnorm/ReadVariableOp2t
8model/layer_normalization_5/batchnorm/mul/ReadVariableOp8model/layer_normalization_5/batchnorm/mul/ReadVariableOp2l
4model/layer_normalization_6/batchnorm/ReadVariableOp4model/layer_normalization_6/batchnorm/ReadVariableOp2t
8model/layer_normalization_6/batchnorm/mul/ReadVariableOp8model/layer_normalization_6/batchnorm/mul/ReadVariableOp2l
4model/layer_normalization_7/batchnorm/ReadVariableOp4model/layer_normalization_7/batchnorm/ReadVariableOp2t
8model/layer_normalization_7/batchnorm/mul/ReadVariableOp8model/layer_normalization_7/batchnorm/mul/ReadVariableOp2
>model/multi_head_attention/attention_output/add/ReadVariableOp>model/multi_head_attention/attention_output/add/ReadVariableOp2
Hmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpHmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2f
1model/multi_head_attention/key/add/ReadVariableOp1model/multi_head_attention/key/add/ReadVariableOp2z
;model/multi_head_attention/key/einsum/Einsum/ReadVariableOp;model/multi_head_attention/key/einsum/Einsum/ReadVariableOp2j
3model/multi_head_attention/query/add/ReadVariableOp3model/multi_head_attention/query/add/ReadVariableOp2~
=model/multi_head_attention/query/einsum/Einsum/ReadVariableOp=model/multi_head_attention/query/einsum/Einsum/ReadVariableOp2j
3model/multi_head_attention/value/add/ReadVariableOp3model/multi_head_attention/value/add/ReadVariableOp2~
=model/multi_head_attention/value/einsum/Einsum/ReadVariableOp=model/multi_head_attention/value/einsum/Einsum/ReadVariableOp2
@model/multi_head_attention_1/attention_output/add/ReadVariableOp@model/multi_head_attention_1/attention_output/add/ReadVariableOp2
Jmodel/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpJmodel/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp2j
3model/multi_head_attention_1/key/add/ReadVariableOp3model/multi_head_attention_1/key/add/ReadVariableOp2~
=model/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp=model/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp2n
5model/multi_head_attention_1/query/add/ReadVariableOp5model/multi_head_attention_1/query/add/ReadVariableOp2
?model/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp?model/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp2n
5model/multi_head_attention_1/value/add/ReadVariableOp5model/multi_head_attention_1/value/add/ReadVariableOp2
?model/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp?model/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp2
@model/multi_head_attention_2/attention_output/add/ReadVariableOp@model/multi_head_attention_2/attention_output/add/ReadVariableOp2
Jmodel/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpJmodel/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp2j
3model/multi_head_attention_2/key/add/ReadVariableOp3model/multi_head_attention_2/key/add/ReadVariableOp2~
=model/multi_head_attention_2/key/einsum/Einsum/ReadVariableOp=model/multi_head_attention_2/key/einsum/Einsum/ReadVariableOp2n
5model/multi_head_attention_2/query/add/ReadVariableOp5model/multi_head_attention_2/query/add/ReadVariableOp2
?model/multi_head_attention_2/query/einsum/Einsum/ReadVariableOp?model/multi_head_attention_2/query/einsum/Einsum/ReadVariableOp2n
5model/multi_head_attention_2/value/add/ReadVariableOp5model/multi_head_attention_2/value/add/ReadVariableOp2
?model/multi_head_attention_2/value/einsum/Einsum/ReadVariableOp?model/multi_head_attention_2/value/einsum/Einsum/ReadVariableOp2
@model/multi_head_attention_3/attention_output/add/ReadVariableOp@model/multi_head_attention_3/attention_output/add/ReadVariableOp2
Jmodel/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpJmodel/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp2j
3model/multi_head_attention_3/key/add/ReadVariableOp3model/multi_head_attention_3/key/add/ReadVariableOp2~
=model/multi_head_attention_3/key/einsum/Einsum/ReadVariableOp=model/multi_head_attention_3/key/einsum/Einsum/ReadVariableOp2n
5model/multi_head_attention_3/query/add/ReadVariableOp5model/multi_head_attention_3/query/add/ReadVariableOp2
?model/multi_head_attention_3/query/einsum/Einsum/ReadVariableOp?model/multi_head_attention_3/query/einsum/Einsum/ReadVariableOp2n
5model/multi_head_attention_3/value/add/ReadVariableOp5model/multi_head_attention_3/value/add/ReadVariableOp2
?model/multi_head_attention_3/value/einsum/Einsum/ReadVariableOp?model/multi_head_attention_3/value/einsum/Einsum/ReadVariableOp:U Q
,
_output_shapes
:?????????Θ
!
_user_specified_name	input_1


c
D__inference_dropout_2_layer_call_and_return_conditional_losses_17812

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:?????????ΘC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:?????????Θ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????Θt
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????Θn
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:?????????Θ^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:?????????Θ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs


c
D__inference_dropout_6_layer_call_and_return_conditional_losses_14459

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:?????????ΘC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:?????????Θ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????Θt
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????Θn
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:?????????Θ^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:?????????Θ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
ύ

P__inference_layer_normalization_6_layer_call_and_return_conditional_losses_18236

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(r
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:?????????Θ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¬
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θb
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:?????????Θ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θh
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θw
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????Θw
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Θg
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:?????????Θ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
φ

C__inference_conv1d_5_layer_call_and_return_conditional_losses_13987

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θd
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
±
E
)__inference_dropout_3_layer_call_fn_17873

inputs
identity΄
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_13802e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:?????????Θ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
½*
ϊ
Q__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_13889	
query	
valueA
+query_einsum_einsum_readvariableop_resource:23
!query_add_readvariableop_resource:2?
)key_einsum_einsum_readvariableop_resource:21
key_add_readvariableop_resource:2A
+value_einsum_einsum_readvariableop_resource:23
!value_add_readvariableop_resource:2L
6attention_output_einsum_einsum_readvariableop_resource:2:
,attention_output_add_readvariableop_resource:
identity’#attention_output/add/ReadVariableOp’-attention_output/einsum/Einsum/ReadVariableOp’key/add/ReadVariableOp’ key/einsum/Einsum/ReadVariableOp’query/add/ReadVariableOp’"query/einsum/Einsum/ReadVariableOp’value/add/ReadVariableOp’"value/einsum/Einsum/ReadVariableOp
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:2*
dtype0
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0­
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:2*
dtype0
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:2*
dtype0
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΓΠ>d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:?????????Θ2
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:?????????ΘΘ*
equationaecd,abcd->acbep
softmax_2/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:?????????ΘΘx
dropout_11/IdentityIdentitysoftmax_2/Softmax:softmax:0*
T0*1
_output_shapes
:?????????ΘΘ©
einsum_1/EinsumEinsumdropout_11/Identity:output:0value/add:z:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationacbe,aecd->abcd¨
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0Φ
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????Θ*
equationabcd,cde->abe
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0ͺ
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θl
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:?????????ΘΨ
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Θ:?????????Θ: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:?????????Θ

_user_specified_namequery:SO
,
_output_shapes
:?????????Θ

_user_specified_namevalue
Ο

C__inference_conv1d_2_layer_call_and_return_conditional_losses_17868

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????ΘU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:?????????Θf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:?????????Θ
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
φ

C__inference_conv1d_5_layer_call_and_return_conditional_losses_18205

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θd
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
ύ

P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_13848

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(r
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:?????????Θ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¬
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θb
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:?????????Θ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θh
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θw
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????Θw
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Θg
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:?????????Θ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs


c
D__inference_dropout_1_layer_call_and_return_conditional_losses_17609

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:?????????ΘC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:?????????Θ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????Θt
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????Θn
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:?????????Θ^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:?????????Θ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
ϋ

N__inference_layer_normalization_layer_call_and_return_conditional_losses_17378

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(r
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:?????????Θ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¬
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θb
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:?????????Θ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θh
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θw
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????Θw
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Θg
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:?????????Θ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs


c
D__inference_dropout_5_layer_call_and_return_conditional_losses_18181

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:?????????ΘC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:?????????Θ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????Θt
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????Θn
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:?????????Θ^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:?????????Θ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
‘±
Ϋ
@__inference_model_layer_call_and_return_conditional_losses_14204

inputs'
layer_normalization_13513:'
layer_normalization_13515:0
multi_head_attention_13554:2,
multi_head_attention_13556:20
multi_head_attention_13558:2,
multi_head_attention_13560:20
multi_head_attention_13562:2,
multi_head_attention_13564:20
multi_head_attention_13566:2(
multi_head_attention_13568:)
layer_normalization_1_13602:)
layer_normalization_1_13604:"
conv1d_13624:
conv1d_13626:$
conv1d_1_13652:
conv1d_1_13654:)
layer_normalization_2_13681:)
layer_normalization_2_13683:2
multi_head_attention_1_13722:2.
multi_head_attention_1_13724:22
multi_head_attention_1_13726:2.
multi_head_attention_1_13728:22
multi_head_attention_1_13730:2.
multi_head_attention_1_13732:22
multi_head_attention_1_13734:2*
multi_head_attention_1_13736:)
layer_normalization_3_13770:)
layer_normalization_3_13772:$
conv1d_2_13792:
conv1d_2_13794:$
conv1d_3_13820:
conv1d_3_13822:)
layer_normalization_4_13849:)
layer_normalization_4_13851:2
multi_head_attention_2_13890:2.
multi_head_attention_2_13892:22
multi_head_attention_2_13894:2.
multi_head_attention_2_13896:22
multi_head_attention_2_13898:2.
multi_head_attention_2_13900:22
multi_head_attention_2_13902:2*
multi_head_attention_2_13904:)
layer_normalization_5_13938:)
layer_normalization_5_13940:$
conv1d_4_13960:
conv1d_4_13962:$
conv1d_5_13988:
conv1d_5_13990:)
layer_normalization_6_14017:)
layer_normalization_6_14019:2
multi_head_attention_3_14058:2.
multi_head_attention_3_14060:22
multi_head_attention_3_14062:2.
multi_head_attention_3_14064:22
multi_head_attention_3_14066:2.
multi_head_attention_3_14068:22
multi_head_attention_3_14070:2*
multi_head_attention_3_14072:)
layer_normalization_7_14106:)
layer_normalization_7_14108:$
conv1d_6_14128:
conv1d_6_14130:$
conv1d_7_14156:
conv1d_7_14158:
dense_14175:
Θ
dense_14177:	 
dense_1_14198:	
dense_1_14200:
identity’conv1d/StatefulPartitionedCall’ conv1d_1/StatefulPartitionedCall’ conv1d_2/StatefulPartitionedCall’ conv1d_3/StatefulPartitionedCall’ conv1d_4/StatefulPartitionedCall’ conv1d_5/StatefulPartitionedCall’ conv1d_6/StatefulPartitionedCall’ conv1d_7/StatefulPartitionedCall’dense/StatefulPartitionedCall’dense_1/StatefulPartitionedCall’+layer_normalization/StatefulPartitionedCall’-layer_normalization_1/StatefulPartitionedCall’-layer_normalization_2/StatefulPartitionedCall’-layer_normalization_3/StatefulPartitionedCall’-layer_normalization_4/StatefulPartitionedCall’-layer_normalization_5/StatefulPartitionedCall’-layer_normalization_6/StatefulPartitionedCall’-layer_normalization_7/StatefulPartitionedCall’,multi_head_attention/StatefulPartitionedCall’.multi_head_attention_1/StatefulPartitionedCall’.multi_head_attention_2/StatefulPartitionedCall’.multi_head_attention_3/StatefulPartitionedCall
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCallinputslayer_normalization_13513layer_normalization_13515*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_13512»
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:04layer_normalization/StatefulPartitionedCall:output:0multi_head_attention_13554multi_head_attention_13556multi_head_attention_13558multi_head_attention_13560multi_head_attention_13562multi_head_attention_13564multi_head_attention_13566multi_head_attention_13568*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_13553ι
dropout/PartitionedCallPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_13576
tf.__operators__.add/AddV2AddV2 dropout/PartitionedCall:output:0inputs*
T0*,
_output_shapes
:?????????ΘΎ
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0layer_normalization_1_13602layer_normalization_1_13604*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_13601
conv1d/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0conv1d_13624conv1d_13626*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_13623ί
dropout_1/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_13634
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conv1d_1_13652conv1d_1_13654*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_13651§
tf.__operators__.add_1/AddV2AddV2)conv1d_1/StatefulPartitionedCall:output:0tf.__operators__.add/AddV2:z:0*
T0*,
_output_shapes
:?????????Θΐ
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0layer_normalization_2_13681layer_normalization_2_13683*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_13680Σ
.multi_head_attention_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:06layer_normalization_2/StatefulPartitionedCall:output:0multi_head_attention_1_13722multi_head_attention_1_13724multi_head_attention_1_13726multi_head_attention_1_13728multi_head_attention_1_13730multi_head_attention_1_13732multi_head_attention_1_13734multi_head_attention_1_13736*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_13721ο
dropout_2/PartitionedCallPartitionedCall7multi_head_attention_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_13744’
tf.__operators__.add_2/AddV2AddV2"dropout_2/PartitionedCall:output:0 tf.__operators__.add_1/AddV2:z:0*
T0*,
_output_shapes
:?????????Θΐ
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0layer_normalization_3_13770layer_normalization_3_13772*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_13769’
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_3/StatefulPartitionedCall:output:0conv1d_2_13792conv1d_2_13794*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_13791α
dropout_3/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_13802
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0conv1d_3_13820conv1d_3_13822*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_3_layer_call_and_return_conditional_losses_13819©
tf.__operators__.add_3/AddV2AddV2)conv1d_3/StatefulPartitionedCall:output:0 tf.__operators__.add_2/AddV2:z:0*
T0*,
_output_shapes
:?????????Θΐ
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_3/AddV2:z:0layer_normalization_4_13849layer_normalization_4_13851*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_13848Σ
.multi_head_attention_2/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:06layer_normalization_4/StatefulPartitionedCall:output:0multi_head_attention_2_13890multi_head_attention_2_13892multi_head_attention_2_13894multi_head_attention_2_13896multi_head_attention_2_13898multi_head_attention_2_13900multi_head_attention_2_13902multi_head_attention_2_13904*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_13889ο
dropout_4/PartitionedCallPartitionedCall7multi_head_attention_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_13912’
tf.__operators__.add_4/AddV2AddV2"dropout_4/PartitionedCall:output:0 tf.__operators__.add_3/AddV2:z:0*
T0*,
_output_shapes
:?????????Θΐ
-layer_normalization_5/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_4/AddV2:z:0layer_normalization_5_13938layer_normalization_5_13940*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_13937’
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_5/StatefulPartitionedCall:output:0conv1d_4_13960conv1d_4_13962*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_4_layer_call_and_return_conditional_losses_13959α
dropout_5/PartitionedCallPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_13970
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0conv1d_5_13988conv1d_5_13990*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_5_layer_call_and_return_conditional_losses_13987©
tf.__operators__.add_5/AddV2AddV2)conv1d_5/StatefulPartitionedCall:output:0 tf.__operators__.add_4/AddV2:z:0*
T0*,
_output_shapes
:?????????Θΐ
-layer_normalization_6/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_5/AddV2:z:0layer_normalization_6_14017layer_normalization_6_14019*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_layer_normalization_6_layer_call_and_return_conditional_losses_14016Σ
.multi_head_attention_3/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_6/StatefulPartitionedCall:output:06layer_normalization_6/StatefulPartitionedCall:output:0multi_head_attention_3_14058multi_head_attention_3_14060multi_head_attention_3_14062multi_head_attention_3_14064multi_head_attention_3_14066multi_head_attention_3_14068multi_head_attention_3_14070multi_head_attention_3_14072*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_multi_head_attention_3_layer_call_and_return_conditional_losses_14057ο
dropout_6/PartitionedCallPartitionedCall7multi_head_attention_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_14080’
tf.__operators__.add_6/AddV2AddV2"dropout_6/PartitionedCall:output:0 tf.__operators__.add_5/AddV2:z:0*
T0*,
_output_shapes
:?????????Θΐ
-layer_normalization_7/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_6/AddV2:z:0layer_normalization_7_14106layer_normalization_7_14108*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_layer_normalization_7_layer_call_and_return_conditional_losses_14105’
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_7/StatefulPartitionedCall:output:0conv1d_6_14128conv1d_6_14130*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_6_layer_call_and_return_conditional_losses_14127α
dropout_7/PartitionedCallPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_14138
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0conv1d_7_14156conv1d_7_14158*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_7_layer_call_and_return_conditional_losses_14155©
tf.__operators__.add_7/AddV2AddV2)conv1d_7/StatefulPartitionedCall:output:0 tf.__operators__.add_6/AddV2:z:0*
T0*,
_output_shapes
:?????????Θς
(global_average_pooling1d/PartitionedCallPartitionedCall tf.__operators__.add_7/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_13480
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_14175dense_14177*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_14174Ϊ
dropout_8/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_14185
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0dense_1_14198dense_1_14200*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_14197w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????ή
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall.^layer_normalization_3/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall.^layer_normalization_5/StatefulPartitionedCall.^layer_normalization_6/StatefulPartitionedCall.^layer_normalization_7/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall/^multi_head_attention_1/StatefulPartitionedCall/^multi_head_attention_2/StatefulPartitionedCall/^multi_head_attention_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*΅
_input_shapes£
 :?????????Θ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2^
-layer_normalization_3/StatefulPartitionedCall-layer_normalization_3/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall2^
-layer_normalization_5/StatefulPartitionedCall-layer_normalization_5/StatefulPartitionedCall2^
-layer_normalization_6/StatefulPartitionedCall-layer_normalization_6/StatefulPartitionedCall2^
-layer_normalization_7/StatefulPartitionedCall-layer_normalization_7/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall2`
.multi_head_attention_1/StatefulPartitionedCall.multi_head_attention_1/StatefulPartitionedCall2`
.multi_head_attention_2/StatefulPartitionedCall.multi_head_attention_2/StatefulPartitionedCall2`
.multi_head_attention_3/StatefulPartitionedCall.multi_head_attention_3/StatefulPartitionedCall:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs


a
B__inference_dropout_layer_call_and_return_conditional_losses_14978

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:?????????ΘC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:?????????Θ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????Θt
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????Θn
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:?????????Θ^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:?????????Θ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
φ

C__inference_conv1d_7_layer_call_and_return_conditional_losses_18491

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θd
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
±
E
)__inference_dropout_2_layer_call_fn_17790

inputs
identity΄
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_13744e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:?????????Θ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
κ

5__inference_layer_normalization_3_layer_call_fn_17821

inputs
unknown:
	unknown_0:
identity’StatefulPartitionedCallκ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_13769t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
Ο

C__inference_conv1d_2_layer_call_and_return_conditional_losses_13791

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????ΘU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:?????????Θf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:?????????Θ
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
κ

5__inference_layer_normalization_4_layer_call_fn_17928

inputs
unknown:
	unknown_0:
identity’StatefulPartitionedCallκ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_13848t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
φ

C__inference_conv1d_7_layer_call_and_return_conditional_losses_14155

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θd
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
φ

C__inference_conv1d_3_layer_call_and_return_conditional_losses_17919

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θd
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
ύ

P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_17557

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(r
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:?????????Θ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¬
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θb
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:?????????Θ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θh
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θw
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????Θw
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Θg
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:?????????Θ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
Ο

C__inference_conv1d_6_layer_call_and_return_conditional_losses_14127

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????ΘU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:?????????Θf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:?????????Θ
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs

λ
6__inference_multi_head_attention_3_layer_call_fn_18258	
query	
value
unknown:2
	unknown_0:2
	unknown_1:2
	unknown_2:2
	unknown_3:2
	unknown_4:2
	unknown_5:2
	unknown_6:
identity’StatefulPartitionedCallΐ
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_multi_head_attention_3_layer_call_and_return_conditional_losses_14057t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Θ:?????????Θ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:?????????Θ

_user_specified_namequery:SO
,
_output_shapes
:?????????Θ

_user_specified_namevalue

λ
6__inference_multi_head_attention_1_layer_call_fn_17708	
query	
value
unknown:2
	unknown_0:2
	unknown_1:2
	unknown_2:2
	unknown_3:2
	unknown_4:2
	unknown_5:2
	unknown_6:
identity’StatefulPartitionedCallΐ
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_14876t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Θ:?????????Θ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:?????????Θ

_user_specified_namequery:SO
,
_output_shapes
:?????????Θ

_user_specified_namevalue
ύ

P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_13769

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(r
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:?????????Θ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¬
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θb
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:?????????Θ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θh
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θw
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????Θw
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Θg
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:?????????Θ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
ϊ
΅E
__inference__traced_save_19026
file_prefix8
4savev2_layer_normalization_gamma_read_readvariableop7
3savev2_layer_normalization_beta_read_readvariableop:
6savev2_layer_normalization_1_gamma_read_readvariableop9
5savev2_layer_normalization_1_beta_read_readvariableop,
(savev2_conv1d_kernel_read_readvariableop*
&savev2_conv1d_bias_read_readvariableop.
*savev2_conv1d_1_kernel_read_readvariableop,
(savev2_conv1d_1_bias_read_readvariableop:
6savev2_layer_normalization_2_gamma_read_readvariableop9
5savev2_layer_normalization_2_beta_read_readvariableop:
6savev2_layer_normalization_3_gamma_read_readvariableop9
5savev2_layer_normalization_3_beta_read_readvariableop.
*savev2_conv1d_2_kernel_read_readvariableop,
(savev2_conv1d_2_bias_read_readvariableop.
*savev2_conv1d_3_kernel_read_readvariableop,
(savev2_conv1d_3_bias_read_readvariableop:
6savev2_layer_normalization_4_gamma_read_readvariableop9
5savev2_layer_normalization_4_beta_read_readvariableop:
6savev2_layer_normalization_5_gamma_read_readvariableop9
5savev2_layer_normalization_5_beta_read_readvariableop.
*savev2_conv1d_4_kernel_read_readvariableop,
(savev2_conv1d_4_bias_read_readvariableop.
*savev2_conv1d_5_kernel_read_readvariableop,
(savev2_conv1d_5_bias_read_readvariableop:
6savev2_layer_normalization_6_gamma_read_readvariableop9
5savev2_layer_normalization_6_beta_read_readvariableop:
6savev2_layer_normalization_7_gamma_read_readvariableop9
5savev2_layer_normalization_7_beta_read_readvariableop.
*savev2_conv1d_6_kernel_read_readvariableop,
(savev2_conv1d_6_bias_read_readvariableop.
*savev2_conv1d_7_kernel_read_readvariableop,
(savev2_conv1d_7_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop@
<savev2_multi_head_attention_query_kernel_read_readvariableop>
:savev2_multi_head_attention_query_bias_read_readvariableop>
:savev2_multi_head_attention_key_kernel_read_readvariableop<
8savev2_multi_head_attention_key_bias_read_readvariableop@
<savev2_multi_head_attention_value_kernel_read_readvariableop>
:savev2_multi_head_attention_value_bias_read_readvariableopK
Gsavev2_multi_head_attention_attention_output_kernel_read_readvariableopI
Esavev2_multi_head_attention_attention_output_bias_read_readvariableopB
>savev2_multi_head_attention_1_query_kernel_read_readvariableop@
<savev2_multi_head_attention_1_query_bias_read_readvariableop@
<savev2_multi_head_attention_1_key_kernel_read_readvariableop>
:savev2_multi_head_attention_1_key_bias_read_readvariableopB
>savev2_multi_head_attention_1_value_kernel_read_readvariableop@
<savev2_multi_head_attention_1_value_bias_read_readvariableopM
Isavev2_multi_head_attention_1_attention_output_kernel_read_readvariableopK
Gsavev2_multi_head_attention_1_attention_output_bias_read_readvariableopB
>savev2_multi_head_attention_2_query_kernel_read_readvariableop@
<savev2_multi_head_attention_2_query_bias_read_readvariableop@
<savev2_multi_head_attention_2_key_kernel_read_readvariableop>
:savev2_multi_head_attention_2_key_bias_read_readvariableopB
>savev2_multi_head_attention_2_value_kernel_read_readvariableop@
<savev2_multi_head_attention_2_value_bias_read_readvariableopM
Isavev2_multi_head_attention_2_attention_output_kernel_read_readvariableopK
Gsavev2_multi_head_attention_2_attention_output_bias_read_readvariableopB
>savev2_multi_head_attention_3_query_kernel_read_readvariableop@
<savev2_multi_head_attention_3_query_bias_read_readvariableop@
<savev2_multi_head_attention_3_key_kernel_read_readvariableop>
:savev2_multi_head_attention_3_key_bias_read_readvariableopB
>savev2_multi_head_attention_3_value_kernel_read_readvariableop@
<savev2_multi_head_attention_3_value_bias_read_readvariableopM
Isavev2_multi_head_attention_3_attention_output_kernel_read_readvariableopK
Gsavev2_multi_head_attention_3_attention_output_bias_read_readvariableop#
savev2_iter_read_readvariableop	$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop'
#savev2_momentum_read_readvariableop"
savev2_rho_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop<
8savev2_layer_normalization_gamma_rms_read_readvariableop;
7savev2_layer_normalization_beta_rms_read_readvariableop>
:savev2_layer_normalization_1_gamma_rms_read_readvariableop=
9savev2_layer_normalization_1_beta_rms_read_readvariableop0
,savev2_conv1d_kernel_rms_read_readvariableop.
*savev2_conv1d_bias_rms_read_readvariableop2
.savev2_conv1d_1_kernel_rms_read_readvariableop0
,savev2_conv1d_1_bias_rms_read_readvariableop>
:savev2_layer_normalization_2_gamma_rms_read_readvariableop=
9savev2_layer_normalization_2_beta_rms_read_readvariableop>
:savev2_layer_normalization_3_gamma_rms_read_readvariableop=
9savev2_layer_normalization_3_beta_rms_read_readvariableop2
.savev2_conv1d_2_kernel_rms_read_readvariableop0
,savev2_conv1d_2_bias_rms_read_readvariableop2
.savev2_conv1d_3_kernel_rms_read_readvariableop0
,savev2_conv1d_3_bias_rms_read_readvariableop>
:savev2_layer_normalization_4_gamma_rms_read_readvariableop=
9savev2_layer_normalization_4_beta_rms_read_readvariableop>
:savev2_layer_normalization_5_gamma_rms_read_readvariableop=
9savev2_layer_normalization_5_beta_rms_read_readvariableop2
.savev2_conv1d_4_kernel_rms_read_readvariableop0
,savev2_conv1d_4_bias_rms_read_readvariableop2
.savev2_conv1d_5_kernel_rms_read_readvariableop0
,savev2_conv1d_5_bias_rms_read_readvariableop>
:savev2_layer_normalization_6_gamma_rms_read_readvariableop=
9savev2_layer_normalization_6_beta_rms_read_readvariableop>
:savev2_layer_normalization_7_gamma_rms_read_readvariableop=
9savev2_layer_normalization_7_beta_rms_read_readvariableop2
.savev2_conv1d_6_kernel_rms_read_readvariableop0
,savev2_conv1d_6_bias_rms_read_readvariableop2
.savev2_conv1d_7_kernel_rms_read_readvariableop0
,savev2_conv1d_7_bias_rms_read_readvariableop/
+savev2_dense_kernel_rms_read_readvariableop-
)savev2_dense_bias_rms_read_readvariableop1
-savev2_dense_1_kernel_rms_read_readvariableop/
+savev2_dense_1_bias_rms_read_readvariableopD
@savev2_multi_head_attention_query_kernel_rms_read_readvariableopB
>savev2_multi_head_attention_query_bias_rms_read_readvariableopB
>savev2_multi_head_attention_key_kernel_rms_read_readvariableop@
<savev2_multi_head_attention_key_bias_rms_read_readvariableopD
@savev2_multi_head_attention_value_kernel_rms_read_readvariableopB
>savev2_multi_head_attention_value_bias_rms_read_readvariableopO
Ksavev2_multi_head_attention_attention_output_kernel_rms_read_readvariableopM
Isavev2_multi_head_attention_attention_output_bias_rms_read_readvariableopF
Bsavev2_multi_head_attention_1_query_kernel_rms_read_readvariableopD
@savev2_multi_head_attention_1_query_bias_rms_read_readvariableopD
@savev2_multi_head_attention_1_key_kernel_rms_read_readvariableopB
>savev2_multi_head_attention_1_key_bias_rms_read_readvariableopF
Bsavev2_multi_head_attention_1_value_kernel_rms_read_readvariableopD
@savev2_multi_head_attention_1_value_bias_rms_read_readvariableopQ
Msavev2_multi_head_attention_1_attention_output_kernel_rms_read_readvariableopO
Ksavev2_multi_head_attention_1_attention_output_bias_rms_read_readvariableopF
Bsavev2_multi_head_attention_2_query_kernel_rms_read_readvariableopD
@savev2_multi_head_attention_2_query_bias_rms_read_readvariableopD
@savev2_multi_head_attention_2_key_kernel_rms_read_readvariableopB
>savev2_multi_head_attention_2_key_bias_rms_read_readvariableopF
Bsavev2_multi_head_attention_2_value_kernel_rms_read_readvariableopD
@savev2_multi_head_attention_2_value_bias_rms_read_readvariableopQ
Msavev2_multi_head_attention_2_attention_output_kernel_rms_read_readvariableopO
Ksavev2_multi_head_attention_2_attention_output_bias_rms_read_readvariableopF
Bsavev2_multi_head_attention_3_query_kernel_rms_read_readvariableopD
@savev2_multi_head_attention_3_query_bias_rms_read_readvariableopD
@savev2_multi_head_attention_3_key_kernel_rms_read_readvariableopB
>savev2_multi_head_attention_3_key_bias_rms_read_readvariableopF
Bsavev2_multi_head_attention_3_value_kernel_rms_read_readvariableopD
@savev2_multi_head_attention_3_value_bias_rms_read_readvariableopQ
Msavev2_multi_head_attention_3_attention_output_kernel_rms_read_readvariableopO
Ksavev2_multi_head_attention_3_attention_output_bias_rms_read_readvariableop
savev2_const

identity_1’MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: H
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*¨G
valueGBGB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/50/.ATTRIBUTES/VARIABLE_VALUEB'variables/51/.ATTRIBUTES/VARIABLE_VALUEB'variables/52/.ATTRIBUTES/VARIABLE_VALUEB'variables/53/.ATTRIBUTES/VARIABLE_VALUEB'variables/54/.ATTRIBUTES/VARIABLE_VALUEB'variables/55/.ATTRIBUTES/VARIABLE_VALUEB'variables/56/.ATTRIBUTES/VARIABLE_VALUEB'variables/57/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-17/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/8/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/9/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/18/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/19/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/20/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/21/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/22/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/23/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/24/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/25/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/34/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/35/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/36/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/37/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/38/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/39/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/40/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/41/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/50/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/51/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/52/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/53/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/54/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/55/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/56/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/57/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*Ί
value°B­B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ΛB
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_layer_normalization_gamma_read_readvariableop3savev2_layer_normalization_beta_read_readvariableop6savev2_layer_normalization_1_gamma_read_readvariableop5savev2_layer_normalization_1_beta_read_readvariableop(savev2_conv1d_kernel_read_readvariableop&savev2_conv1d_bias_read_readvariableop*savev2_conv1d_1_kernel_read_readvariableop(savev2_conv1d_1_bias_read_readvariableop6savev2_layer_normalization_2_gamma_read_readvariableop5savev2_layer_normalization_2_beta_read_readvariableop6savev2_layer_normalization_3_gamma_read_readvariableop5savev2_layer_normalization_3_beta_read_readvariableop*savev2_conv1d_2_kernel_read_readvariableop(savev2_conv1d_2_bias_read_readvariableop*savev2_conv1d_3_kernel_read_readvariableop(savev2_conv1d_3_bias_read_readvariableop6savev2_layer_normalization_4_gamma_read_readvariableop5savev2_layer_normalization_4_beta_read_readvariableop6savev2_layer_normalization_5_gamma_read_readvariableop5savev2_layer_normalization_5_beta_read_readvariableop*savev2_conv1d_4_kernel_read_readvariableop(savev2_conv1d_4_bias_read_readvariableop*savev2_conv1d_5_kernel_read_readvariableop(savev2_conv1d_5_bias_read_readvariableop6savev2_layer_normalization_6_gamma_read_readvariableop5savev2_layer_normalization_6_beta_read_readvariableop6savev2_layer_normalization_7_gamma_read_readvariableop5savev2_layer_normalization_7_beta_read_readvariableop*savev2_conv1d_6_kernel_read_readvariableop(savev2_conv1d_6_bias_read_readvariableop*savev2_conv1d_7_kernel_read_readvariableop(savev2_conv1d_7_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop<savev2_multi_head_attention_query_kernel_read_readvariableop:savev2_multi_head_attention_query_bias_read_readvariableop:savev2_multi_head_attention_key_kernel_read_readvariableop8savev2_multi_head_attention_key_bias_read_readvariableop<savev2_multi_head_attention_value_kernel_read_readvariableop:savev2_multi_head_attention_value_bias_read_readvariableopGsavev2_multi_head_attention_attention_output_kernel_read_readvariableopEsavev2_multi_head_attention_attention_output_bias_read_readvariableop>savev2_multi_head_attention_1_query_kernel_read_readvariableop<savev2_multi_head_attention_1_query_bias_read_readvariableop<savev2_multi_head_attention_1_key_kernel_read_readvariableop:savev2_multi_head_attention_1_key_bias_read_readvariableop>savev2_multi_head_attention_1_value_kernel_read_readvariableop<savev2_multi_head_attention_1_value_bias_read_readvariableopIsavev2_multi_head_attention_1_attention_output_kernel_read_readvariableopGsavev2_multi_head_attention_1_attention_output_bias_read_readvariableop>savev2_multi_head_attention_2_query_kernel_read_readvariableop<savev2_multi_head_attention_2_query_bias_read_readvariableop<savev2_multi_head_attention_2_key_kernel_read_readvariableop:savev2_multi_head_attention_2_key_bias_read_readvariableop>savev2_multi_head_attention_2_value_kernel_read_readvariableop<savev2_multi_head_attention_2_value_bias_read_readvariableopIsavev2_multi_head_attention_2_attention_output_kernel_read_readvariableopGsavev2_multi_head_attention_2_attention_output_bias_read_readvariableop>savev2_multi_head_attention_3_query_kernel_read_readvariableop<savev2_multi_head_attention_3_query_bias_read_readvariableop<savev2_multi_head_attention_3_key_kernel_read_readvariableop:savev2_multi_head_attention_3_key_bias_read_readvariableop>savev2_multi_head_attention_3_value_kernel_read_readvariableop<savev2_multi_head_attention_3_value_bias_read_readvariableopIsavev2_multi_head_attention_3_attention_output_kernel_read_readvariableopGsavev2_multi_head_attention_3_attention_output_bias_read_readvariableopsavev2_iter_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop#savev2_momentum_read_readvariableopsavev2_rho_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop8savev2_layer_normalization_gamma_rms_read_readvariableop7savev2_layer_normalization_beta_rms_read_readvariableop:savev2_layer_normalization_1_gamma_rms_read_readvariableop9savev2_layer_normalization_1_beta_rms_read_readvariableop,savev2_conv1d_kernel_rms_read_readvariableop*savev2_conv1d_bias_rms_read_readvariableop.savev2_conv1d_1_kernel_rms_read_readvariableop,savev2_conv1d_1_bias_rms_read_readvariableop:savev2_layer_normalization_2_gamma_rms_read_readvariableop9savev2_layer_normalization_2_beta_rms_read_readvariableop:savev2_layer_normalization_3_gamma_rms_read_readvariableop9savev2_layer_normalization_3_beta_rms_read_readvariableop.savev2_conv1d_2_kernel_rms_read_readvariableop,savev2_conv1d_2_bias_rms_read_readvariableop.savev2_conv1d_3_kernel_rms_read_readvariableop,savev2_conv1d_3_bias_rms_read_readvariableop:savev2_layer_normalization_4_gamma_rms_read_readvariableop9savev2_layer_normalization_4_beta_rms_read_readvariableop:savev2_layer_normalization_5_gamma_rms_read_readvariableop9savev2_layer_normalization_5_beta_rms_read_readvariableop.savev2_conv1d_4_kernel_rms_read_readvariableop,savev2_conv1d_4_bias_rms_read_readvariableop.savev2_conv1d_5_kernel_rms_read_readvariableop,savev2_conv1d_5_bias_rms_read_readvariableop:savev2_layer_normalization_6_gamma_rms_read_readvariableop9savev2_layer_normalization_6_beta_rms_read_readvariableop:savev2_layer_normalization_7_gamma_rms_read_readvariableop9savev2_layer_normalization_7_beta_rms_read_readvariableop.savev2_conv1d_6_kernel_rms_read_readvariableop,savev2_conv1d_6_bias_rms_read_readvariableop.savev2_conv1d_7_kernel_rms_read_readvariableop,savev2_conv1d_7_bias_rms_read_readvariableop+savev2_dense_kernel_rms_read_readvariableop)savev2_dense_bias_rms_read_readvariableop-savev2_dense_1_kernel_rms_read_readvariableop+savev2_dense_1_bias_rms_read_readvariableop@savev2_multi_head_attention_query_kernel_rms_read_readvariableop>savev2_multi_head_attention_query_bias_rms_read_readvariableop>savev2_multi_head_attention_key_kernel_rms_read_readvariableop<savev2_multi_head_attention_key_bias_rms_read_readvariableop@savev2_multi_head_attention_value_kernel_rms_read_readvariableop>savev2_multi_head_attention_value_bias_rms_read_readvariableopKsavev2_multi_head_attention_attention_output_kernel_rms_read_readvariableopIsavev2_multi_head_attention_attention_output_bias_rms_read_readvariableopBsavev2_multi_head_attention_1_query_kernel_rms_read_readvariableop@savev2_multi_head_attention_1_query_bias_rms_read_readvariableop@savev2_multi_head_attention_1_key_kernel_rms_read_readvariableop>savev2_multi_head_attention_1_key_bias_rms_read_readvariableopBsavev2_multi_head_attention_1_value_kernel_rms_read_readvariableop@savev2_multi_head_attention_1_value_bias_rms_read_readvariableopMsavev2_multi_head_attention_1_attention_output_kernel_rms_read_readvariableopKsavev2_multi_head_attention_1_attention_output_bias_rms_read_readvariableopBsavev2_multi_head_attention_2_query_kernel_rms_read_readvariableop@savev2_multi_head_attention_2_query_bias_rms_read_readvariableop@savev2_multi_head_attention_2_key_kernel_rms_read_readvariableop>savev2_multi_head_attention_2_key_bias_rms_read_readvariableopBsavev2_multi_head_attention_2_value_kernel_rms_read_readvariableop@savev2_multi_head_attention_2_value_bias_rms_read_readvariableopMsavev2_multi_head_attention_2_attention_output_kernel_rms_read_readvariableopKsavev2_multi_head_attention_2_attention_output_bias_rms_read_readvariableopBsavev2_multi_head_attention_3_query_kernel_rms_read_readvariableop@savev2_multi_head_attention_3_query_bias_rms_read_readvariableop@savev2_multi_head_attention_3_key_kernel_rms_read_readvariableop>savev2_multi_head_attention_3_key_bias_rms_read_readvariableopBsavev2_multi_head_attention_3_value_kernel_rms_read_readvariableop@savev2_multi_head_attention_3_value_bias_rms_read_readvariableopMsavev2_multi_head_attention_3_attention_output_kernel_rms_read_readvariableopKsavev2_multi_head_attention_3_attention_output_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *£
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Σ

_input_shapesΑ

Ύ
: :::::::::::::::::::::::::::::::::
Θ::	::2:2:2:2:2:2:2::2:2:2:2:2:2:2::2:2:2:2:2:2:2::2:2:2:2:2:2:2:: : : : : : : : : :::::::::::::::::::::::::::::::::
Θ::	::2:2:2:2:2:2:2::2:2:2:2:2:2:2::2:2:2:2:2:2:2::2:2:2:2:2:2:2:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
::  

_output_shapes
::&!"
 
_output_shapes
:
Θ:!"

_output_shapes	
::%#!

_output_shapes
:	: $

_output_shapes
::(%$
"
_output_shapes
:2:$& 

_output_shapes

:2:('$
"
_output_shapes
:2:$( 

_output_shapes

:2:()$
"
_output_shapes
:2:$* 

_output_shapes

:2:(+$
"
_output_shapes
:2: ,

_output_shapes
::(-$
"
_output_shapes
:2:$. 

_output_shapes

:2:(/$
"
_output_shapes
:2:$0 

_output_shapes

:2:(1$
"
_output_shapes
:2:$2 

_output_shapes

:2:(3$
"
_output_shapes
:2: 4

_output_shapes
::(5$
"
_output_shapes
:2:$6 

_output_shapes

:2:(7$
"
_output_shapes
:2:$8 

_output_shapes

:2:(9$
"
_output_shapes
:2:$: 

_output_shapes

:2:(;$
"
_output_shapes
:2: <

_output_shapes
::(=$
"
_output_shapes
:2:$> 

_output_shapes

:2:(?$
"
_output_shapes
:2:$@ 

_output_shapes

:2:(A$
"
_output_shapes
:2:$B 

_output_shapes

:2:(C$
"
_output_shapes
:2: D

_output_shapes
::E

_output_shapes
: :F

_output_shapes
: :G

_output_shapes
: :H

_output_shapes
: :I

_output_shapes
: :J

_output_shapes
: :K

_output_shapes
: :L

_output_shapes
: :M

_output_shapes
: : N

_output_shapes
:: O

_output_shapes
:: P

_output_shapes
:: Q

_output_shapes
::(R$
"
_output_shapes
:: S

_output_shapes
::(T$
"
_output_shapes
:: U

_output_shapes
:: V

_output_shapes
:: W

_output_shapes
:: X

_output_shapes
:: Y

_output_shapes
::(Z$
"
_output_shapes
:: [

_output_shapes
::(\$
"
_output_shapes
:: ]

_output_shapes
:: ^

_output_shapes
:: _

_output_shapes
:: `

_output_shapes
:: a

_output_shapes
::(b$
"
_output_shapes
:: c

_output_shapes
::(d$
"
_output_shapes
:: e

_output_shapes
:: f

_output_shapes
:: g

_output_shapes
:: h

_output_shapes
:: i

_output_shapes
::(j$
"
_output_shapes
:: k

_output_shapes
::(l$
"
_output_shapes
:: m

_output_shapes
::&n"
 
_output_shapes
:
Θ:!o

_output_shapes	
::%p!

_output_shapes
:	: q

_output_shapes
::(r$
"
_output_shapes
:2:$s 

_output_shapes

:2:(t$
"
_output_shapes
:2:$u 

_output_shapes

:2:(v$
"
_output_shapes
:2:$w 

_output_shapes

:2:(x$
"
_output_shapes
:2: y

_output_shapes
::(z$
"
_output_shapes
:2:${ 

_output_shapes

:2:(|$
"
_output_shapes
:2:$} 

_output_shapes

:2:(~$
"
_output_shapes
:2:$ 

_output_shapes

:2:)$
"
_output_shapes
:2:!

_output_shapes
::)$
"
_output_shapes
:2:% 

_output_shapes

:2:)$
"
_output_shapes
:2:% 

_output_shapes

:2:)$
"
_output_shapes
:2:% 

_output_shapes

:2:)$
"
_output_shapes
:2:!

_output_shapes
::)$
"
_output_shapes
:2:% 

_output_shapes

:2:)$
"
_output_shapes
:2:% 

_output_shapes

:2:)$
"
_output_shapes
:2:% 

_output_shapes

:2:)$
"
_output_shapes
:2:!

_output_shapes
::

_output_shapes
: 
λ
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_17800

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:?????????Θ`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:?????????Θ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs

b
)__inference_dropout_4_layer_call_fn_18081

inputs
identity’StatefulPartitionedCallΔ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_14632t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
ύ

P__inference_layer_normalization_7_layer_call_and_return_conditional_losses_18415

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(r
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:?????????Θ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¬
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θb
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:?????????Θ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θh
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θw
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????Θw
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Θg
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:?????????Θ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs


c
D__inference_dropout_1_layer_call_and_return_conditional_losses_14935

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:?????????ΘC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:?????????Θ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????Θt
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????Θn
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:?????????Θ^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:?????????Θ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs


c
D__inference_dropout_2_layer_call_and_return_conditional_losses_14805

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:?????????ΘC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:?????????Θ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????Θt
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????Θn
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:?????????Θ^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:?????????Θ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
·2
ϊ
Q__inference_multi_head_attention_3_layer_call_and_return_conditional_losses_14530	
query	
valueA
+query_einsum_einsum_readvariableop_resource:23
!query_add_readvariableop_resource:2?
)key_einsum_einsum_readvariableop_resource:21
key_add_readvariableop_resource:2A
+value_einsum_einsum_readvariableop_resource:23
!value_add_readvariableop_resource:2L
6attention_output_einsum_einsum_readvariableop_resource:2:
,attention_output_add_readvariableop_resource:
identity’#attention_output/add/ReadVariableOp’-attention_output/einsum/Einsum/ReadVariableOp’key/add/ReadVariableOp’ key/einsum/Einsum/ReadVariableOp’query/add/ReadVariableOp’"query/einsum/Einsum/ReadVariableOp’value/add/ReadVariableOp’"value/einsum/Einsum/ReadVariableOp
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:2*
dtype0
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0­
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:2*
dtype0
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:2*
dtype0
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΓΠ>d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:?????????Θ2
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:?????????ΘΘ*
equationaecd,abcd->acbep
softmax_3/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:?????????ΘΘ]
dropout_12/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ?
dropout_12/dropout/MulMulsoftmax_3/Softmax:softmax:0!dropout_12/dropout/Const:output:0*
T0*1
_output_shapes
:?????????ΘΘc
dropout_12/dropout/ShapeShapesoftmax_3/Softmax:softmax:0*
T0*
_output_shapes
:¬
/dropout_12/dropout/random_uniform/RandomUniformRandomUniform!dropout_12/dropout/Shape:output:0*
T0*1
_output_shapes
:?????????ΘΘ*
dtype0f
!dropout_12/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >Ρ
dropout_12/dropout/GreaterEqualGreaterEqual8dropout_12/dropout/random_uniform/RandomUniform:output:0*dropout_12/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:?????????ΘΘ
dropout_12/dropout/CastCast#dropout_12/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:?????????ΘΘ
dropout_12/dropout/Mul_1Muldropout_12/dropout/Mul:z:0dropout_12/dropout/Cast:y:0*
T0*1
_output_shapes
:?????????ΘΘ©
einsum_1/EinsumEinsumdropout_12/dropout/Mul_1:z:0value/add:z:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationacbe,aecd->abcd¨
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0Φ
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????Θ*
equationabcd,cde->abe
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0ͺ
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θl
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:?????????ΘΨ
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Θ:?????????Θ: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:?????????Θ

_user_specified_namequery:SO
,
_output_shapes
:?????????Θ

_user_specified_namevalue
Ψ

(__inference_conv1d_7_layer_call_fn_18476

inputs
unknown:
	unknown_0:
identity’StatefulPartitionedCallέ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_7_layer_call_and_return_conditional_losses_14155t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
λ
b
D__inference_dropout_7_layer_call_and_return_conditional_losses_14138

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:?????????Θ`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:?????????Θ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs

λ
6__inference_multi_head_attention_2_layer_call_fn_17972	
query	
value
unknown:2
	unknown_0:2
	unknown_1:2
	unknown_2:2
	unknown_3:2
	unknown_4:2
	unknown_5:2
	unknown_6:
identity’StatefulPartitionedCallΐ
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_13889t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Θ:?????????Θ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:?????????Θ

_user_specified_namequery:SO
,
_output_shapes
:?????????Θ

_user_specified_namevalue

o
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_18502

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
Ο

C__inference_conv1d_4_layer_call_and_return_conditional_losses_18154

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????ΘU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:?????????Θf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:?????????Θ
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
ύ

P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_13601

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(r
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:?????????Θ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¬
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θb
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:?????????Θ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θh
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θw
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????Θw
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Θg
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:?????????Θ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
Ο

C__inference_conv1d_4_layer_call_and_return_conditional_losses_13959

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????ΘU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:?????????Θf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:?????????Θ
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
½*
ϊ
Q__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_13721	
query	
valueA
+query_einsum_einsum_readvariableop_resource:23
!query_add_readvariableop_resource:2?
)key_einsum_einsum_readvariableop_resource:21
key_add_readvariableop_resource:2A
+value_einsum_einsum_readvariableop_resource:23
!value_add_readvariableop_resource:2L
6attention_output_einsum_einsum_readvariableop_resource:2:
,attention_output_add_readvariableop_resource:
identity’#attention_output/add/ReadVariableOp’-attention_output/einsum/Einsum/ReadVariableOp’key/add/ReadVariableOp’ key/einsum/Einsum/ReadVariableOp’query/add/ReadVariableOp’"query/einsum/Einsum/ReadVariableOp’value/add/ReadVariableOp’"value/einsum/Einsum/ReadVariableOp
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:2*
dtype0
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0­
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:2*
dtype0
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:2*
dtype0
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΓΠ>d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:?????????Θ2
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:?????????ΘΘ*
equationaecd,abcd->acbep
softmax_1/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:?????????ΘΘx
dropout_10/IdentityIdentitysoftmax_1/Softmax:softmax:0*
T0*1
_output_shapes
:?????????ΘΘ©
einsum_1/EinsumEinsumdropout_10/Identity:output:0value/add:z:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationacbe,aecd->abcd¨
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0Φ
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????Θ*
equationabcd,cde->abe
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0ͺ
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θl
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:?????????ΘΨ
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Θ:?????????Θ: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:?????????Θ

_user_specified_namequery:SO
,
_output_shapes
:?????????Θ

_user_specified_namevalue

΅
%__inference_model_layer_call_fn_16472

inputs
unknown:
	unknown_0:
	unknown_1:2
	unknown_2:2
	unknown_3:2
	unknown_4:2
	unknown_5:2
	unknown_6:2
	unknown_7:2
	unknown_8:
	unknown_9:

unknown_10: 

unknown_11:

unknown_12: 

unknown_13:

unknown_14:

unknown_15:

unknown_16: 

unknown_17:2

unknown_18:2 

unknown_19:2

unknown_20:2 

unknown_21:2

unknown_22:2 

unknown_23:2

unknown_24:

unknown_25:

unknown_26: 

unknown_27:

unknown_28: 

unknown_29:

unknown_30:

unknown_31:

unknown_32: 

unknown_33:2

unknown_34:2 

unknown_35:2

unknown_36:2 

unknown_37:2

unknown_38:2 

unknown_39:2

unknown_40:

unknown_41:

unknown_42: 

unknown_43:

unknown_44: 

unknown_45:

unknown_46:

unknown_47:

unknown_48: 

unknown_49:2

unknown_50:2 

unknown_51:2

unknown_52:2 

unknown_53:2

unknown_54:2 

unknown_55:2

unknown_56:

unknown_57:

unknown_58: 

unknown_59:

unknown_60: 

unknown_61:

unknown_62:

unknown_63:
Θ

unknown_64:	

unknown_65:	

unknown_66:
identity’StatefulPartitionedCallθ	
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66*P
TinI
G2E*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*f
_read_only_resource_inputsH
FD	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCD*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_15401o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*΅
_input_shapes£
 :?????????Θ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
2
ψ
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_15049	
query	
valueA
+query_einsum_einsum_readvariableop_resource:23
!query_add_readvariableop_resource:2?
)key_einsum_einsum_readvariableop_resource:21
key_add_readvariableop_resource:2A
+value_einsum_einsum_readvariableop_resource:23
!value_add_readvariableop_resource:2L
6attention_output_einsum_einsum_readvariableop_resource:2:
,attention_output_add_readvariableop_resource:
identity’#attention_output/add/ReadVariableOp’-attention_output/einsum/Einsum/ReadVariableOp’key/add/ReadVariableOp’ key/einsum/Einsum/ReadVariableOp’query/add/ReadVariableOp’"query/einsum/Einsum/ReadVariableOp’value/add/ReadVariableOp’"value/einsum/Einsum/ReadVariableOp
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:2*
dtype0
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0­
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:2*
dtype0
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:2*
dtype0
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΓΠ>d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:?????????Θ2
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:?????????ΘΘ*
equationaecd,abcd->acben
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:?????????ΘΘ\
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ?
dropout_9/dropout/MulMulsoftmax/Softmax:softmax:0 dropout_9/dropout/Const:output:0*
T0*1
_output_shapes
:?????????ΘΘ`
dropout_9/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:ͺ
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*1
_output_shapes
:?????????ΘΘ*
dtype0e
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >Ξ
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:?????????ΘΘ
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:?????????ΘΘ
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*1
_output_shapes
:?????????ΘΘ¨
einsum_1/EinsumEinsumdropout_9/dropout/Mul_1:z:0value/add:z:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationacbe,aecd->abcd¨
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0Φ
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????Θ*
equationabcd,cde->abe
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0ͺ
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θl
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:?????????ΘΨ
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Θ:?????????Θ: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:?????????Θ

_user_specified_namequery:SO
,
_output_shapes
:?????????Θ

_user_specified_namevalue
λ
b
D__inference_dropout_6_layer_call_and_return_conditional_losses_14080

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:?????????Θ`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:?????????Θ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
ϊ	
c
D__inference_dropout_8_layer_call_and_return_conditional_losses_14373

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUΥ?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ>§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:?????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
½*
ϊ
Q__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_17743	
query	
valueA
+query_einsum_einsum_readvariableop_resource:23
!query_add_readvariableop_resource:2?
)key_einsum_einsum_readvariableop_resource:21
key_add_readvariableop_resource:2A
+value_einsum_einsum_readvariableop_resource:23
!value_add_readvariableop_resource:2L
6attention_output_einsum_einsum_readvariableop_resource:2:
,attention_output_add_readvariableop_resource:
identity’#attention_output/add/ReadVariableOp’-attention_output/einsum/Einsum/ReadVariableOp’key/add/ReadVariableOp’ key/einsum/Einsum/ReadVariableOp’query/add/ReadVariableOp’"query/einsum/Einsum/ReadVariableOp’value/add/ReadVariableOp’"value/einsum/Einsum/ReadVariableOp
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:2*
dtype0
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0­
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:2*
dtype0
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:2*
dtype0
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΓΠ>d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:?????????Θ2
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:?????????ΘΘ*
equationaecd,abcd->acbep
softmax_1/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:?????????ΘΘx
dropout_10/IdentityIdentitysoftmax_1/Softmax:softmax:0*
T0*1
_output_shapes
:?????????ΘΘ©
einsum_1/EinsumEinsumdropout_10/Identity:output:0value/add:z:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationacbe,aecd->abcd¨
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0Φ
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????Θ*
equationabcd,cde->abe
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0ͺ
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θl
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:?????????ΘΨ
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Θ:?????????Θ: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:?????????Θ

_user_specified_namequery:SO
,
_output_shapes
:?????????Θ

_user_specified_namevalue
ΜΚ
―C
@__inference_model_layer_call_and_return_conditional_losses_16864

inputsG
9layer_normalization_batchnorm_mul_readvariableop_resource:C
5layer_normalization_batchnorm_readvariableop_resource:V
@multi_head_attention_query_einsum_einsum_readvariableop_resource:2H
6multi_head_attention_query_add_readvariableop_resource:2T
>multi_head_attention_key_einsum_einsum_readvariableop_resource:2F
4multi_head_attention_key_add_readvariableop_resource:2V
@multi_head_attention_value_einsum_einsum_readvariableop_resource:2H
6multi_head_attention_value_add_readvariableop_resource:2a
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource:2O
Amulti_head_attention_attention_output_add_readvariableop_resource:I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:E
7layer_normalization_1_batchnorm_readvariableop_resource:H
2conv1d_conv1d_expanddims_1_readvariableop_resource:4
&conv1d_biasadd_readvariableop_resource:J
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_1_biasadd_readvariableop_resource:I
;layer_normalization_2_batchnorm_mul_readvariableop_resource:E
7layer_normalization_2_batchnorm_readvariableop_resource:X
Bmulti_head_attention_1_query_einsum_einsum_readvariableop_resource:2J
8multi_head_attention_1_query_add_readvariableop_resource:2V
@multi_head_attention_1_key_einsum_einsum_readvariableop_resource:2H
6multi_head_attention_1_key_add_readvariableop_resource:2X
Bmulti_head_attention_1_value_einsum_einsum_readvariableop_resource:2J
8multi_head_attention_1_value_add_readvariableop_resource:2c
Mmulti_head_attention_1_attention_output_einsum_einsum_readvariableop_resource:2Q
Cmulti_head_attention_1_attention_output_add_readvariableop_resource:I
;layer_normalization_3_batchnorm_mul_readvariableop_resource:E
7layer_normalization_3_batchnorm_readvariableop_resource:J
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_2_biasadd_readvariableop_resource:J
4conv1d_3_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_3_biasadd_readvariableop_resource:I
;layer_normalization_4_batchnorm_mul_readvariableop_resource:E
7layer_normalization_4_batchnorm_readvariableop_resource:X
Bmulti_head_attention_2_query_einsum_einsum_readvariableop_resource:2J
8multi_head_attention_2_query_add_readvariableop_resource:2V
@multi_head_attention_2_key_einsum_einsum_readvariableop_resource:2H
6multi_head_attention_2_key_add_readvariableop_resource:2X
Bmulti_head_attention_2_value_einsum_einsum_readvariableop_resource:2J
8multi_head_attention_2_value_add_readvariableop_resource:2c
Mmulti_head_attention_2_attention_output_einsum_einsum_readvariableop_resource:2Q
Cmulti_head_attention_2_attention_output_add_readvariableop_resource:I
;layer_normalization_5_batchnorm_mul_readvariableop_resource:E
7layer_normalization_5_batchnorm_readvariableop_resource:J
4conv1d_4_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_4_biasadd_readvariableop_resource:J
4conv1d_5_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_5_biasadd_readvariableop_resource:I
;layer_normalization_6_batchnorm_mul_readvariableop_resource:E
7layer_normalization_6_batchnorm_readvariableop_resource:X
Bmulti_head_attention_3_query_einsum_einsum_readvariableop_resource:2J
8multi_head_attention_3_query_add_readvariableop_resource:2V
@multi_head_attention_3_key_einsum_einsum_readvariableop_resource:2H
6multi_head_attention_3_key_add_readvariableop_resource:2X
Bmulti_head_attention_3_value_einsum_einsum_readvariableop_resource:2J
8multi_head_attention_3_value_add_readvariableop_resource:2c
Mmulti_head_attention_3_attention_output_einsum_einsum_readvariableop_resource:2Q
Cmulti_head_attention_3_attention_output_add_readvariableop_resource:I
;layer_normalization_7_batchnorm_mul_readvariableop_resource:E
7layer_normalization_7_batchnorm_readvariableop_resource:J
4conv1d_6_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_6_biasadd_readvariableop_resource:J
4conv1d_7_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_7_biasadd_readvariableop_resource:8
$dense_matmul_readvariableop_resource:
Θ4
%dense_biasadd_readvariableop_resource:	9
&dense_1_matmul_readvariableop_resource:	5
'dense_1_biasadd_readvariableop_resource:
identity’conv1d/BiasAdd/ReadVariableOp’)conv1d/Conv1D/ExpandDims_1/ReadVariableOp’conv1d_1/BiasAdd/ReadVariableOp’+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp’conv1d_2/BiasAdd/ReadVariableOp’+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp’conv1d_3/BiasAdd/ReadVariableOp’+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp’conv1d_4/BiasAdd/ReadVariableOp’+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp’conv1d_5/BiasAdd/ReadVariableOp’+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp’conv1d_6/BiasAdd/ReadVariableOp’+conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp’conv1d_7/BiasAdd/ReadVariableOp’+conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp’dense/BiasAdd/ReadVariableOp’dense/MatMul/ReadVariableOp’dense_1/BiasAdd/ReadVariableOp’dense_1/MatMul/ReadVariableOp’,layer_normalization/batchnorm/ReadVariableOp’0layer_normalization/batchnorm/mul/ReadVariableOp’.layer_normalization_1/batchnorm/ReadVariableOp’2layer_normalization_1/batchnorm/mul/ReadVariableOp’.layer_normalization_2/batchnorm/ReadVariableOp’2layer_normalization_2/batchnorm/mul/ReadVariableOp’.layer_normalization_3/batchnorm/ReadVariableOp’2layer_normalization_3/batchnorm/mul/ReadVariableOp’.layer_normalization_4/batchnorm/ReadVariableOp’2layer_normalization_4/batchnorm/mul/ReadVariableOp’.layer_normalization_5/batchnorm/ReadVariableOp’2layer_normalization_5/batchnorm/mul/ReadVariableOp’.layer_normalization_6/batchnorm/ReadVariableOp’2layer_normalization_6/batchnorm/mul/ReadVariableOp’.layer_normalization_7/batchnorm/ReadVariableOp’2layer_normalization_7/batchnorm/mul/ReadVariableOp’8multi_head_attention/attention_output/add/ReadVariableOp’Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp’+multi_head_attention/key/add/ReadVariableOp’5multi_head_attention/key/einsum/Einsum/ReadVariableOp’-multi_head_attention/query/add/ReadVariableOp’7multi_head_attention/query/einsum/Einsum/ReadVariableOp’-multi_head_attention/value/add/ReadVariableOp’7multi_head_attention/value/einsum/Einsum/ReadVariableOp’:multi_head_attention_1/attention_output/add/ReadVariableOp’Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp’-multi_head_attention_1/key/add/ReadVariableOp’7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp’/multi_head_attention_1/query/add/ReadVariableOp’9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp’/multi_head_attention_1/value/add/ReadVariableOp’9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp’:multi_head_attention_2/attention_output/add/ReadVariableOp’Dmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp’-multi_head_attention_2/key/add/ReadVariableOp’7multi_head_attention_2/key/einsum/Einsum/ReadVariableOp’/multi_head_attention_2/query/add/ReadVariableOp’9multi_head_attention_2/query/einsum/Einsum/ReadVariableOp’/multi_head_attention_2/value/add/ReadVariableOp’9multi_head_attention_2/value/einsum/Einsum/ReadVariableOp’:multi_head_attention_3/attention_output/add/ReadVariableOp’Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp’-multi_head_attention_3/key/add/ReadVariableOp’7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp’/multi_head_attention_3/query/add/ReadVariableOp’9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp’/multi_head_attention_3/value/add/ReadVariableOp’9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:΅
 layer_normalization/moments/meanMeaninputs;layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:?????????Θ΄
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceinputs1layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θ
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:θ
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Ύ
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θ
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:?????????Θ¦
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Β
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ
#layer_normalization/batchnorm/mul_1Mulinputs%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θ³
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θ
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Ύ
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????Θ³
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????ΘΌ
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0ύ
(multi_head_attention/query/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde€
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:2*
dtype0Μ
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2Έ
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0ω
&multi_head_attention/key/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde 
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:2*
dtype0Ζ
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2Ό
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0ύ
(multi_head_attention/value/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde€
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:2*
dtype0Μ
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2_
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΓΠ>£
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:?????????Θ2Π
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*1
_output_shapes
:?????????ΘΘ*
equationaecd,abcd->acbe
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*1
_output_shapes
:?????????ΘΘ
'multi_head_attention/dropout_9/IdentityIdentity.multi_head_attention/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:?????????ΘΘη
$multi_head_attention/einsum_1/EinsumEinsum0multi_head_attention/dropout_9/Identity:output:0"multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationacbe,aecd->abcd?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????Θ*
equationabcd,cde->abeΆ
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0ι
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ
dropout/IdentityIdentity-multi_head_attention/attention_output/add:z:0*
T0*,
_output_shapes
:?????????Θ}
tf.__operators__.add/AddV2AddV2dropout/Identity:output:0inputs*
T0*,
_output_shapes
:?????????Θ~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Ρ
"layer_normalization_1/moments/meanMeantf.__operators__.add/AddV2:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:?????????ΘΠ
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferencetf.__operators__.add/AddV2:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θ
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ξ
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Δ
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θ
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:?????????Θͺ
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Θ
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ¬
%layer_normalization_1/batchnorm/mul_1Multf.__operators__.add/AddV2:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ΘΉ
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θ’
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Δ
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????ΘΉ
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Θg
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????³
conv1d/Conv1D/ExpandDims
ExpandDims)layer_normalization_1/batchnorm/add_1:z:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ 
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ΅
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Γ
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θc
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:?????????Θp
dropout_1/IdentityIdentityconv1d/Relu:activations:0*
T0*,
_output_shapes
:?????????Θi
conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????©
conv1d_1/Conv1D/ExpandDims
ExpandDimsdropout_1/Identity:output:0'conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ€
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_1/Conv1D/ExpandDims_1
ExpandDims3conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Ι
conv1d_1/Conv1DConv2D#conv1d_1/Conv1D/ExpandDims:output:0%conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

conv1d_1/Conv1D/SqueezeSqueezeconv1d_1/Conv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv1d_1/BiasAddBiasAdd conv1d_1/Conv1D/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ
tf.__operators__.add_1/AddV2AddV2conv1d_1/BiasAdd:output:0tf.__operators__.add/AddV2:z:0*
T0*,
_output_shapes
:?????????Θ~
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Σ
"layer_normalization_2/moments/meanMean tf.__operators__.add_1/AddV2:z:0=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*,
_output_shapes
:?????????Θ?
/layer_normalization_2/moments/SquaredDifferenceSquaredDifference tf.__operators__.add_1/AddV2:z:03layer_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θ
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ξ
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(j
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Δ
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θ
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*,
_output_shapes
:?????????Θͺ
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Θ
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ?
%layer_normalization_2/batchnorm/mul_1Mul tf.__operators__.add_1/AddV2:z:0'layer_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ΘΉ
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θ’
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Δ
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????ΘΉ
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Θΐ
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0
*multi_head_attention_1/query/einsum/EinsumEinsum)layer_normalization_2/batchnorm/add_1:z:0Amulti_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde¨
/multi_head_attention_1/query/add/ReadVariableOpReadVariableOp8multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

:2*
dtype0?
 multi_head_attention_1/query/addAddV23multi_head_attention_1/query/einsum/Einsum:output:07multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2Ό
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0?
(multi_head_attention_1/key/einsum/EinsumEinsum)layer_normalization_2/batchnorm/add_1:z:0?multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde€
-multi_head_attention_1/key/add/ReadVariableOpReadVariableOp6multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

:2*
dtype0Μ
multi_head_attention_1/key/addAddV21multi_head_attention_1/key/einsum/Einsum:output:05multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2ΐ
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0
*multi_head_attention_1/value/einsum/EinsumEinsum)layer_normalization_2/batchnorm/add_1:z:0Amulti_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde¨
/multi_head_attention_1/value/add/ReadVariableOpReadVariableOp8multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

:2*
dtype0?
 multi_head_attention_1/value/addAddV23multi_head_attention_1/value/einsum/Einsum:output:07multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2a
multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΓΠ>©
multi_head_attention_1/MulMul$multi_head_attention_1/query/add:z:0%multi_head_attention_1/Mul/y:output:0*
T0*0
_output_shapes
:?????????Θ2Φ
$multi_head_attention_1/einsum/EinsumEinsum"multi_head_attention_1/key/add:z:0multi_head_attention_1/Mul:z:0*
N*
T0*1
_output_shapes
:?????????ΘΘ*
equationaecd,abcd->acbe
(multi_head_attention_1/softmax_1/SoftmaxSoftmax-multi_head_attention_1/einsum/Einsum:output:0*
T0*1
_output_shapes
:?????????ΘΘ¦
*multi_head_attention_1/dropout_10/IdentityIdentity2multi_head_attention_1/softmax_1/Softmax:softmax:0*
T0*1
_output_shapes
:?????????ΘΘξ
&multi_head_attention_1/einsum_1/EinsumEinsum3multi_head_attention_1/dropout_10/Identity:output:0$multi_head_attention_1/value/add:z:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationacbe,aecd->abcdΦ
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0
5multi_head_attention_1/attention_output/einsum/EinsumEinsum/multi_head_attention_1/einsum_1/Einsum:output:0Lmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????Θ*
equationabcd,cde->abeΊ
:multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0ο
+multi_head_attention_1/attention_output/addAddV2>multi_head_attention_1/attention_output/einsum/Einsum:output:0Bmulti_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ
dropout_2/IdentityIdentity/multi_head_attention_1/attention_output/add:z:0*
T0*,
_output_shapes
:?????????Θ
tf.__operators__.add_2/AddV2AddV2dropout_2/Identity:output:0 tf.__operators__.add_1/AddV2:z:0*
T0*,
_output_shapes
:?????????Θ~
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Σ
"layer_normalization_3/moments/meanMean tf.__operators__.add_2/AddV2:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*,
_output_shapes
:?????????Θ?
/layer_normalization_3/moments/SquaredDifferenceSquaredDifference tf.__operators__.add_2/AddV2:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θ
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ξ
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(j
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Δ
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θ
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*,
_output_shapes
:?????????Θͺ
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Θ
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ?
%layer_normalization_3/batchnorm/mul_1Mul tf.__operators__.add_2/AddV2:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ΘΉ
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θ’
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Δ
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????ΘΉ
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Θi
conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????·
conv1d_2/Conv1D/ExpandDims
ExpandDims)layer_normalization_3/batchnorm/add_1:z:0'conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ€
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_2/Conv1D/ExpandDims_1
ExpandDims3conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Ι
conv1d_2/Conv1DConv2D#conv1d_2/Conv1D/ExpandDims:output:0%conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

conv1d_2/Conv1D/SqueezeSqueezeconv1d_2/Conv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv1d_2/BiasAddBiasAdd conv1d_2/Conv1D/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θg
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*,
_output_shapes
:?????????Θr
dropout_3/IdentityIdentityconv1d_2/Relu:activations:0*
T0*,
_output_shapes
:?????????Θi
conv1d_3/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????©
conv1d_3/Conv1D/ExpandDims
ExpandDimsdropout_3/Identity:output:0'conv1d_3/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ€
+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_3/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_3/Conv1D/ExpandDims_1
ExpandDims3conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Ι
conv1d_3/Conv1DConv2D#conv1d_3/Conv1D/ExpandDims:output:0%conv1d_3/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

conv1d_3/Conv1D/SqueezeSqueezeconv1d_3/Conv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv1d_3/BiasAddBiasAdd conv1d_3/Conv1D/Squeeze:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ
tf.__operators__.add_3/AddV2AddV2conv1d_3/BiasAdd:output:0 tf.__operators__.add_2/AddV2:z:0*
T0*,
_output_shapes
:?????????Θ~
4layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Σ
"layer_normalization_4/moments/meanMean tf.__operators__.add_3/AddV2:z:0=layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(
*layer_normalization_4/moments/StopGradientStopGradient+layer_normalization_4/moments/mean:output:0*
T0*,
_output_shapes
:?????????Θ?
/layer_normalization_4/moments/SquaredDifferenceSquaredDifference tf.__operators__.add_3/AddV2:z:03layer_normalization_4/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θ
8layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ξ
&layer_normalization_4/moments/varianceMean3layer_normalization_4/moments/SquaredDifference:z:0Alayer_normalization_4/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(j
%layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Δ
#layer_normalization_4/batchnorm/addAddV2/layer_normalization_4/moments/variance:output:0.layer_normalization_4/batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θ
%layer_normalization_4/batchnorm/RsqrtRsqrt'layer_normalization_4/batchnorm/add:z:0*
T0*,
_output_shapes
:?????????Θͺ
2layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Θ
#layer_normalization_4/batchnorm/mulMul)layer_normalization_4/batchnorm/Rsqrt:y:0:layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ?
%layer_normalization_4/batchnorm/mul_1Mul tf.__operators__.add_3/AddV2:z:0'layer_normalization_4/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ΘΉ
%layer_normalization_4/batchnorm/mul_2Mul+layer_normalization_4/moments/mean:output:0'layer_normalization_4/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θ’
.layer_normalization_4/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Δ
#layer_normalization_4/batchnorm/subSub6layer_normalization_4/batchnorm/ReadVariableOp:value:0)layer_normalization_4/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????ΘΉ
%layer_normalization_4/batchnorm/add_1AddV2)layer_normalization_4/batchnorm/mul_1:z:0'layer_normalization_4/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Θΐ
9multi_head_attention_2/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_2_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0
*multi_head_attention_2/query/einsum/EinsumEinsum)layer_normalization_4/batchnorm/add_1:z:0Amulti_head_attention_2/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde¨
/multi_head_attention_2/query/add/ReadVariableOpReadVariableOp8multi_head_attention_2_query_add_readvariableop_resource*
_output_shapes

:2*
dtype0?
 multi_head_attention_2/query/addAddV23multi_head_attention_2/query/einsum/Einsum:output:07multi_head_attention_2/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2Ό
7multi_head_attention_2/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_2_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0?
(multi_head_attention_2/key/einsum/EinsumEinsum)layer_normalization_4/batchnorm/add_1:z:0?multi_head_attention_2/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde€
-multi_head_attention_2/key/add/ReadVariableOpReadVariableOp6multi_head_attention_2_key_add_readvariableop_resource*
_output_shapes

:2*
dtype0Μ
multi_head_attention_2/key/addAddV21multi_head_attention_2/key/einsum/Einsum:output:05multi_head_attention_2/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2ΐ
9multi_head_attention_2/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_2_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0
*multi_head_attention_2/value/einsum/EinsumEinsum)layer_normalization_4/batchnorm/add_1:z:0Amulti_head_attention_2/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde¨
/multi_head_attention_2/value/add/ReadVariableOpReadVariableOp8multi_head_attention_2_value_add_readvariableop_resource*
_output_shapes

:2*
dtype0?
 multi_head_attention_2/value/addAddV23multi_head_attention_2/value/einsum/Einsum:output:07multi_head_attention_2/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2a
multi_head_attention_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΓΠ>©
multi_head_attention_2/MulMul$multi_head_attention_2/query/add:z:0%multi_head_attention_2/Mul/y:output:0*
T0*0
_output_shapes
:?????????Θ2Φ
$multi_head_attention_2/einsum/EinsumEinsum"multi_head_attention_2/key/add:z:0multi_head_attention_2/Mul:z:0*
N*
T0*1
_output_shapes
:?????????ΘΘ*
equationaecd,abcd->acbe
(multi_head_attention_2/softmax_2/SoftmaxSoftmax-multi_head_attention_2/einsum/Einsum:output:0*
T0*1
_output_shapes
:?????????ΘΘ¦
*multi_head_attention_2/dropout_11/IdentityIdentity2multi_head_attention_2/softmax_2/Softmax:softmax:0*
T0*1
_output_shapes
:?????????ΘΘξ
&multi_head_attention_2/einsum_1/EinsumEinsum3multi_head_attention_2/dropout_11/Identity:output:0$multi_head_attention_2/value/add:z:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationacbe,aecd->abcdΦ
Dmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_2_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0
5multi_head_attention_2/attention_output/einsum/EinsumEinsum/multi_head_attention_2/einsum_1/Einsum:output:0Lmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????Θ*
equationabcd,cde->abeΊ
:multi_head_attention_2/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_2_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0ο
+multi_head_attention_2/attention_output/addAddV2>multi_head_attention_2/attention_output/einsum/Einsum:output:0Bmulti_head_attention_2/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ
dropout_4/IdentityIdentity/multi_head_attention_2/attention_output/add:z:0*
T0*,
_output_shapes
:?????????Θ
tf.__operators__.add_4/AddV2AddV2dropout_4/Identity:output:0 tf.__operators__.add_3/AddV2:z:0*
T0*,
_output_shapes
:?????????Θ~
4layer_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Σ
"layer_normalization_5/moments/meanMean tf.__operators__.add_4/AddV2:z:0=layer_normalization_5/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(
*layer_normalization_5/moments/StopGradientStopGradient+layer_normalization_5/moments/mean:output:0*
T0*,
_output_shapes
:?????????Θ?
/layer_normalization_5/moments/SquaredDifferenceSquaredDifference tf.__operators__.add_4/AddV2:z:03layer_normalization_5/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θ
8layer_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ξ
&layer_normalization_5/moments/varianceMean3layer_normalization_5/moments/SquaredDifference:z:0Alayer_normalization_5/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(j
%layer_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Δ
#layer_normalization_5/batchnorm/addAddV2/layer_normalization_5/moments/variance:output:0.layer_normalization_5/batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θ
%layer_normalization_5/batchnorm/RsqrtRsqrt'layer_normalization_5/batchnorm/add:z:0*
T0*,
_output_shapes
:?????????Θͺ
2layer_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Θ
#layer_normalization_5/batchnorm/mulMul)layer_normalization_5/batchnorm/Rsqrt:y:0:layer_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ?
%layer_normalization_5/batchnorm/mul_1Mul tf.__operators__.add_4/AddV2:z:0'layer_normalization_5/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ΘΉ
%layer_normalization_5/batchnorm/mul_2Mul+layer_normalization_5/moments/mean:output:0'layer_normalization_5/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θ’
.layer_normalization_5/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Δ
#layer_normalization_5/batchnorm/subSub6layer_normalization_5/batchnorm/ReadVariableOp:value:0)layer_normalization_5/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????ΘΉ
%layer_normalization_5/batchnorm/add_1AddV2)layer_normalization_5/batchnorm/mul_1:z:0'layer_normalization_5/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Θi
conv1d_4/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????·
conv1d_4/Conv1D/ExpandDims
ExpandDims)layer_normalization_5/batchnorm/add_1:z:0'conv1d_4/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ€
+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_4/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_4/Conv1D/ExpandDims_1
ExpandDims3conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_4/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Ι
conv1d_4/Conv1DConv2D#conv1d_4/Conv1D/ExpandDims:output:0%conv1d_4/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

conv1d_4/Conv1D/SqueezeSqueezeconv1d_4/Conv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????
conv1d_4/BiasAdd/ReadVariableOpReadVariableOp(conv1d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv1d_4/BiasAddBiasAdd conv1d_4/Conv1D/Squeeze:output:0'conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θg
conv1d_4/ReluReluconv1d_4/BiasAdd:output:0*
T0*,
_output_shapes
:?????????Θr
dropout_5/IdentityIdentityconv1d_4/Relu:activations:0*
T0*,
_output_shapes
:?????????Θi
conv1d_5/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????©
conv1d_5/Conv1D/ExpandDims
ExpandDimsdropout_5/Identity:output:0'conv1d_5/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ€
+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_5/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_5/Conv1D/ExpandDims_1
ExpandDims3conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_5/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Ι
conv1d_5/Conv1DConv2D#conv1d_5/Conv1D/ExpandDims:output:0%conv1d_5/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

conv1d_5/Conv1D/SqueezeSqueezeconv1d_5/Conv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????
conv1d_5/BiasAdd/ReadVariableOpReadVariableOp(conv1d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv1d_5/BiasAddBiasAdd conv1d_5/Conv1D/Squeeze:output:0'conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ
tf.__operators__.add_5/AddV2AddV2conv1d_5/BiasAdd:output:0 tf.__operators__.add_4/AddV2:z:0*
T0*,
_output_shapes
:?????????Θ~
4layer_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Σ
"layer_normalization_6/moments/meanMean tf.__operators__.add_5/AddV2:z:0=layer_normalization_6/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(
*layer_normalization_6/moments/StopGradientStopGradient+layer_normalization_6/moments/mean:output:0*
T0*,
_output_shapes
:?????????Θ?
/layer_normalization_6/moments/SquaredDifferenceSquaredDifference tf.__operators__.add_5/AddV2:z:03layer_normalization_6/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θ
8layer_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ξ
&layer_normalization_6/moments/varianceMean3layer_normalization_6/moments/SquaredDifference:z:0Alayer_normalization_6/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(j
%layer_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Δ
#layer_normalization_6/batchnorm/addAddV2/layer_normalization_6/moments/variance:output:0.layer_normalization_6/batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θ
%layer_normalization_6/batchnorm/RsqrtRsqrt'layer_normalization_6/batchnorm/add:z:0*
T0*,
_output_shapes
:?????????Θͺ
2layer_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Θ
#layer_normalization_6/batchnorm/mulMul)layer_normalization_6/batchnorm/Rsqrt:y:0:layer_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ?
%layer_normalization_6/batchnorm/mul_1Mul tf.__operators__.add_5/AddV2:z:0'layer_normalization_6/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ΘΉ
%layer_normalization_6/batchnorm/mul_2Mul+layer_normalization_6/moments/mean:output:0'layer_normalization_6/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θ’
.layer_normalization_6/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Δ
#layer_normalization_6/batchnorm/subSub6layer_normalization_6/batchnorm/ReadVariableOp:value:0)layer_normalization_6/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????ΘΉ
%layer_normalization_6/batchnorm/add_1AddV2)layer_normalization_6/batchnorm/mul_1:z:0'layer_normalization_6/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Θΐ
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_3_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0
*multi_head_attention_3/query/einsum/EinsumEinsum)layer_normalization_6/batchnorm/add_1:z:0Amulti_head_attention_3/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde¨
/multi_head_attention_3/query/add/ReadVariableOpReadVariableOp8multi_head_attention_3_query_add_readvariableop_resource*
_output_shapes

:2*
dtype0?
 multi_head_attention_3/query/addAddV23multi_head_attention_3/query/einsum/Einsum:output:07multi_head_attention_3/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2Ό
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_3_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0?
(multi_head_attention_3/key/einsum/EinsumEinsum)layer_normalization_6/batchnorm/add_1:z:0?multi_head_attention_3/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde€
-multi_head_attention_3/key/add/ReadVariableOpReadVariableOp6multi_head_attention_3_key_add_readvariableop_resource*
_output_shapes

:2*
dtype0Μ
multi_head_attention_3/key/addAddV21multi_head_attention_3/key/einsum/Einsum:output:05multi_head_attention_3/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2ΐ
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_3_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0
*multi_head_attention_3/value/einsum/EinsumEinsum)layer_normalization_6/batchnorm/add_1:z:0Amulti_head_attention_3/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde¨
/multi_head_attention_3/value/add/ReadVariableOpReadVariableOp8multi_head_attention_3_value_add_readvariableop_resource*
_output_shapes

:2*
dtype0?
 multi_head_attention_3/value/addAddV23multi_head_attention_3/value/einsum/Einsum:output:07multi_head_attention_3/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2a
multi_head_attention_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΓΠ>©
multi_head_attention_3/MulMul$multi_head_attention_3/query/add:z:0%multi_head_attention_3/Mul/y:output:0*
T0*0
_output_shapes
:?????????Θ2Φ
$multi_head_attention_3/einsum/EinsumEinsum"multi_head_attention_3/key/add:z:0multi_head_attention_3/Mul:z:0*
N*
T0*1
_output_shapes
:?????????ΘΘ*
equationaecd,abcd->acbe
(multi_head_attention_3/softmax_3/SoftmaxSoftmax-multi_head_attention_3/einsum/Einsum:output:0*
T0*1
_output_shapes
:?????????ΘΘ¦
*multi_head_attention_3/dropout_12/IdentityIdentity2multi_head_attention_3/softmax_3/Softmax:softmax:0*
T0*1
_output_shapes
:?????????ΘΘξ
&multi_head_attention_3/einsum_1/EinsumEinsum3multi_head_attention_3/dropout_12/Identity:output:0$multi_head_attention_3/value/add:z:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationacbe,aecd->abcdΦ
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_3_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0
5multi_head_attention_3/attention_output/einsum/EinsumEinsum/multi_head_attention_3/einsum_1/Einsum:output:0Lmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????Θ*
equationabcd,cde->abeΊ
:multi_head_attention_3/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_3_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0ο
+multi_head_attention_3/attention_output/addAddV2>multi_head_attention_3/attention_output/einsum/Einsum:output:0Bmulti_head_attention_3/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ
dropout_6/IdentityIdentity/multi_head_attention_3/attention_output/add:z:0*
T0*,
_output_shapes
:?????????Θ
tf.__operators__.add_6/AddV2AddV2dropout_6/Identity:output:0 tf.__operators__.add_5/AddV2:z:0*
T0*,
_output_shapes
:?????????Θ~
4layer_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Σ
"layer_normalization_7/moments/meanMean tf.__operators__.add_6/AddV2:z:0=layer_normalization_7/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(
*layer_normalization_7/moments/StopGradientStopGradient+layer_normalization_7/moments/mean:output:0*
T0*,
_output_shapes
:?????????Θ?
/layer_normalization_7/moments/SquaredDifferenceSquaredDifference tf.__operators__.add_6/AddV2:z:03layer_normalization_7/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θ
8layer_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ξ
&layer_normalization_7/moments/varianceMean3layer_normalization_7/moments/SquaredDifference:z:0Alayer_normalization_7/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(j
%layer_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Δ
#layer_normalization_7/batchnorm/addAddV2/layer_normalization_7/moments/variance:output:0.layer_normalization_7/batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θ
%layer_normalization_7/batchnorm/RsqrtRsqrt'layer_normalization_7/batchnorm/add:z:0*
T0*,
_output_shapes
:?????????Θͺ
2layer_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Θ
#layer_normalization_7/batchnorm/mulMul)layer_normalization_7/batchnorm/Rsqrt:y:0:layer_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ?
%layer_normalization_7/batchnorm/mul_1Mul tf.__operators__.add_6/AddV2:z:0'layer_normalization_7/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ΘΉ
%layer_normalization_7/batchnorm/mul_2Mul+layer_normalization_7/moments/mean:output:0'layer_normalization_7/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θ’
.layer_normalization_7/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Δ
#layer_normalization_7/batchnorm/subSub6layer_normalization_7/batchnorm/ReadVariableOp:value:0)layer_normalization_7/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????ΘΉ
%layer_normalization_7/batchnorm/add_1AddV2)layer_normalization_7/batchnorm/mul_1:z:0'layer_normalization_7/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Θi
conv1d_6/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????·
conv1d_6/Conv1D/ExpandDims
ExpandDims)layer_normalization_7/batchnorm/add_1:z:0'conv1d_6/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ€
+conv1d_6/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_6/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_6/Conv1D/ExpandDims_1
ExpandDims3conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_6/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Ι
conv1d_6/Conv1DConv2D#conv1d_6/Conv1D/ExpandDims:output:0%conv1d_6/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

conv1d_6/Conv1D/SqueezeSqueezeconv1d_6/Conv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????
conv1d_6/BiasAdd/ReadVariableOpReadVariableOp(conv1d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv1d_6/BiasAddBiasAdd conv1d_6/Conv1D/Squeeze:output:0'conv1d_6/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θg
conv1d_6/ReluReluconv1d_6/BiasAdd:output:0*
T0*,
_output_shapes
:?????????Θr
dropout_7/IdentityIdentityconv1d_6/Relu:activations:0*
T0*,
_output_shapes
:?????????Θi
conv1d_7/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????©
conv1d_7/Conv1D/ExpandDims
ExpandDimsdropout_7/Identity:output:0'conv1d_7/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ€
+conv1d_7/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_7_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_7/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_7/Conv1D/ExpandDims_1
ExpandDims3conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_7/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Ι
conv1d_7/Conv1DConv2D#conv1d_7/Conv1D/ExpandDims:output:0%conv1d_7/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

conv1d_7/Conv1D/SqueezeSqueezeconv1d_7/Conv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????
conv1d_7/BiasAdd/ReadVariableOpReadVariableOp(conv1d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv1d_7/BiasAddBiasAdd conv1d_7/Conv1D/Squeeze:output:0'conv1d_7/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ
tf.__operators__.add_7/AddV2AddV2conv1d_7/BiasAdd:output:0 tf.__operators__.add_6/AddV2:z:0*
T0*,
_output_shapes
:?????????Θq
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :΄
global_average_pooling1d/MeanMean tf.__operators__.add_7/AddV2:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:?????????Θ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
Θ*
dtype0
dense/MatMulMatMul&global_average_pooling1d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????k
dropout_8/IdentityIdentitydense/Relu:activations:0*
T0*(
_output_shapes
:?????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_1/MatMulMatMuldropout_8/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_3/BiasAdd/ReadVariableOp,^conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_4/BiasAdd/ReadVariableOp,^conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_5/BiasAdd/ReadVariableOp,^conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_6/BiasAdd/ReadVariableOp,^conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_7/BiasAdd/ReadVariableOp,^conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp/^layer_normalization_4/batchnorm/ReadVariableOp3^layer_normalization_4/batchnorm/mul/ReadVariableOp/^layer_normalization_5/batchnorm/ReadVariableOp3^layer_normalization_5/batchnorm/mul/ReadVariableOp/^layer_normalization_6/batchnorm/ReadVariableOp3^layer_normalization_6/batchnorm/mul/ReadVariableOp/^layer_normalization_7/batchnorm/ReadVariableOp3^layer_normalization_7/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp;^multi_head_attention_1/attention_output/add/ReadVariableOpE^multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_1/key/add/ReadVariableOp8^multi_head_attention_1/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/query/add/ReadVariableOp:^multi_head_attention_1/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/value/add/ReadVariableOp:^multi_head_attention_1/value/einsum/Einsum/ReadVariableOp;^multi_head_attention_2/attention_output/add/ReadVariableOpE^multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_2/key/add/ReadVariableOp8^multi_head_attention_2/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_2/query/add/ReadVariableOp:^multi_head_attention_2/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_2/value/add/ReadVariableOp:^multi_head_attention_2/value/einsum/Einsum/ReadVariableOp;^multi_head_attention_3/attention_output/add/ReadVariableOpE^multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_3/key/add/ReadVariableOp8^multi_head_attention_3/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_3/query/add/ReadVariableOp:^multi_head_attention_3/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_3/value/add/ReadVariableOp:^multi_head_attention_3/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*΅
_input_shapes£
 :?????????Θ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2Z
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_3/BiasAdd/ReadVariableOpconv1d_3/BiasAdd/ReadVariableOp2Z
+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_4/BiasAdd/ReadVariableOpconv1d_4/BiasAdd/ReadVariableOp2Z
+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_5/BiasAdd/ReadVariableOpconv1d_5/BiasAdd/ReadVariableOp2Z
+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_6/BiasAdd/ReadVariableOpconv1d_6/BiasAdd/ReadVariableOp2Z
+conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_7/BiasAdd/ReadVariableOpconv1d_7/BiasAdd/ReadVariableOp2Z
+conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2`
.layer_normalization_2/batchnorm/ReadVariableOp.layer_normalization_2/batchnorm/ReadVariableOp2h
2layer_normalization_2/batchnorm/mul/ReadVariableOp2layer_normalization_2/batchnorm/mul/ReadVariableOp2`
.layer_normalization_3/batchnorm/ReadVariableOp.layer_normalization_3/batchnorm/ReadVariableOp2h
2layer_normalization_3/batchnorm/mul/ReadVariableOp2layer_normalization_3/batchnorm/mul/ReadVariableOp2`
.layer_normalization_4/batchnorm/ReadVariableOp.layer_normalization_4/batchnorm/ReadVariableOp2h
2layer_normalization_4/batchnorm/mul/ReadVariableOp2layer_normalization_4/batchnorm/mul/ReadVariableOp2`
.layer_normalization_5/batchnorm/ReadVariableOp.layer_normalization_5/batchnorm/ReadVariableOp2h
2layer_normalization_5/batchnorm/mul/ReadVariableOp2layer_normalization_5/batchnorm/mul/ReadVariableOp2`
.layer_normalization_6/batchnorm/ReadVariableOp.layer_normalization_6/batchnorm/ReadVariableOp2h
2layer_normalization_6/batchnorm/mul/ReadVariableOp2layer_normalization_6/batchnorm/mul/ReadVariableOp2`
.layer_normalization_7/batchnorm/ReadVariableOp.layer_normalization_7/batchnorm/ReadVariableOp2h
2layer_normalization_7/batchnorm/mul/ReadVariableOp2layer_normalization_7/batchnorm/mul/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp2x
:multi_head_attention_1/attention_output/add/ReadVariableOp:multi_head_attention_1/attention_output/add/ReadVariableOp2
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_1/key/add/ReadVariableOp-multi_head_attention_1/key/add/ReadVariableOp2r
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_1/query/add/ReadVariableOp/multi_head_attention_1/query/add/ReadVariableOp2v
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_1/value/add/ReadVariableOp/multi_head_attention_1/value/add/ReadVariableOp2v
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp2x
:multi_head_attention_2/attention_output/add/ReadVariableOp:multi_head_attention_2/attention_output/add/ReadVariableOp2
Dmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_2/key/add/ReadVariableOp-multi_head_attention_2/key/add/ReadVariableOp2r
7multi_head_attention_2/key/einsum/Einsum/ReadVariableOp7multi_head_attention_2/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_2/query/add/ReadVariableOp/multi_head_attention_2/query/add/ReadVariableOp2v
9multi_head_attention_2/query/einsum/Einsum/ReadVariableOp9multi_head_attention_2/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_2/value/add/ReadVariableOp/multi_head_attention_2/value/add/ReadVariableOp2v
9multi_head_attention_2/value/einsum/Einsum/ReadVariableOp9multi_head_attention_2/value/einsum/Einsum/ReadVariableOp2x
:multi_head_attention_3/attention_output/add/ReadVariableOp:multi_head_attention_3/attention_output/add/ReadVariableOp2
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_3/key/add/ReadVariableOp-multi_head_attention_3/key/add/ReadVariableOp2r
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_3/query/add/ReadVariableOp/multi_head_attention_3/query/add/ReadVariableOp2v
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_3/value/add/ReadVariableOp/multi_head_attention_3/value/add/ReadVariableOp2v
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
Ψ

(__inference_conv1d_2_layer_call_fn_17852

inputs
unknown:
	unknown_0:
identity’StatefulPartitionedCallέ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_13791t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
ύ

P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_17664

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(r
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:?????????Θ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¬
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θb
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:?????????Θ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θh
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θw
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????Θw
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Θg
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:?????????Θ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
£

τ
@__inference_dense_layer_call_and_return_conditional_losses_18522

inputs2
matmul_readvariableop_resource:
Θ.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Θ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
ύ

P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_17843

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(r
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:?????????Θ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¬
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θb
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:?????????Θ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θh
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θw
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????Θw
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Θg
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:?????????Θ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs

λ
6__inference_multi_head_attention_2_layer_call_fn_17994	
query	
value
unknown:2
	unknown_0:2
	unknown_1:2
	unknown_2:2
	unknown_3:2
	unknown_4:2
	unknown_5:2
	unknown_6:
identity’StatefulPartitionedCallΐ
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_14703t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Θ:?????????Θ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:?????????Θ

_user_specified_namequery:SO
,
_output_shapes
:?????????Θ

_user_specified_namevalue

λ
6__inference_multi_head_attention_3_layer_call_fn_18280	
query	
value
unknown:2
	unknown_0:2
	unknown_1:2
	unknown_2:2
	unknown_3:2
	unknown_4:2
	unknown_5:2
	unknown_6:
identity’StatefulPartitionedCallΐ
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_multi_head_attention_3_layer_call_and_return_conditional_losses_14530t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Θ:?????????Θ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:?????????Θ

_user_specified_namequery:SO
,
_output_shapes
:?????????Θ

_user_specified_namevalue
σ
b
)__inference_dropout_8_layer_call_fn_18532

inputs
identity’StatefulPartitionedCallΐ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_14373p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ύ

P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_13680

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(r
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:?????????Θ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¬
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θb
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:?????????Θ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θh
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θw
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????Θw
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Θg
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:?????????Θ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs

Ά
%__inference_model_layer_call_fn_15681
input_1
unknown:
	unknown_0:
	unknown_1:2
	unknown_2:2
	unknown_3:2
	unknown_4:2
	unknown_5:2
	unknown_6:2
	unknown_7:2
	unknown_8:
	unknown_9:

unknown_10: 

unknown_11:

unknown_12: 

unknown_13:

unknown_14:

unknown_15:

unknown_16: 

unknown_17:2

unknown_18:2 

unknown_19:2

unknown_20:2 

unknown_21:2

unknown_22:2 

unknown_23:2

unknown_24:

unknown_25:

unknown_26: 

unknown_27:

unknown_28: 

unknown_29:

unknown_30:

unknown_31:

unknown_32: 

unknown_33:2

unknown_34:2 

unknown_35:2

unknown_36:2 

unknown_37:2

unknown_38:2 

unknown_39:2

unknown_40:

unknown_41:

unknown_42: 

unknown_43:

unknown_44: 

unknown_45:

unknown_46:

unknown_47:

unknown_48: 

unknown_49:2

unknown_50:2 

unknown_51:2

unknown_52:2 

unknown_53:2

unknown_54:2 

unknown_55:2

unknown_56:

unknown_57:

unknown_58: 

unknown_59:

unknown_60: 

unknown_61:

unknown_62:

unknown_63:
Θ

unknown_64:	

unknown_65:	

unknown_66:
identity’StatefulPartitionedCallι	
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66*P
TinI
G2E*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*f
_read_only_resource_inputsH
FD	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCD*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_15401o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*΅
_input_shapes£
 :?????????Θ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:?????????Θ
!
_user_specified_name	input_1
ω
T
8__inference_global_average_pooling1d_layer_call_fn_18496

inputs
identityΗ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_13480i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
£Ώ
!
@__inference_model_layer_call_and_return_conditional_losses_16041
input_1'
layer_normalization_15864:'
layer_normalization_15866:0
multi_head_attention_15869:2,
multi_head_attention_15871:20
multi_head_attention_15873:2,
multi_head_attention_15875:20
multi_head_attention_15877:2,
multi_head_attention_15879:20
multi_head_attention_15881:2(
multi_head_attention_15883:)
layer_normalization_1_15888:)
layer_normalization_1_15890:"
conv1d_15893:
conv1d_15895:$
conv1d_1_15899:
conv1d_1_15901:)
layer_normalization_2_15905:)
layer_normalization_2_15907:2
multi_head_attention_1_15910:2.
multi_head_attention_1_15912:22
multi_head_attention_1_15914:2.
multi_head_attention_1_15916:22
multi_head_attention_1_15918:2.
multi_head_attention_1_15920:22
multi_head_attention_1_15922:2*
multi_head_attention_1_15924:)
layer_normalization_3_15929:)
layer_normalization_3_15931:$
conv1d_2_15934:
conv1d_2_15936:$
conv1d_3_15940:
conv1d_3_15942:)
layer_normalization_4_15946:)
layer_normalization_4_15948:2
multi_head_attention_2_15951:2.
multi_head_attention_2_15953:22
multi_head_attention_2_15955:2.
multi_head_attention_2_15957:22
multi_head_attention_2_15959:2.
multi_head_attention_2_15961:22
multi_head_attention_2_15963:2*
multi_head_attention_2_15965:)
layer_normalization_5_15970:)
layer_normalization_5_15972:$
conv1d_4_15975:
conv1d_4_15977:$
conv1d_5_15981:
conv1d_5_15983:)
layer_normalization_6_15987:)
layer_normalization_6_15989:2
multi_head_attention_3_15992:2.
multi_head_attention_3_15994:22
multi_head_attention_3_15996:2.
multi_head_attention_3_15998:22
multi_head_attention_3_16000:2.
multi_head_attention_3_16002:22
multi_head_attention_3_16004:2*
multi_head_attention_3_16006:)
layer_normalization_7_16011:)
layer_normalization_7_16013:$
conv1d_6_16016:
conv1d_6_16018:$
conv1d_7_16022:
conv1d_7_16024:
dense_16029:
Θ
dense_16031:	 
dense_1_16035:	
dense_1_16037:
identity’conv1d/StatefulPartitionedCall’ conv1d_1/StatefulPartitionedCall’ conv1d_2/StatefulPartitionedCall’ conv1d_3/StatefulPartitionedCall’ conv1d_4/StatefulPartitionedCall’ conv1d_5/StatefulPartitionedCall’ conv1d_6/StatefulPartitionedCall’ conv1d_7/StatefulPartitionedCall’dense/StatefulPartitionedCall’dense_1/StatefulPartitionedCall’dropout/StatefulPartitionedCall’!dropout_1/StatefulPartitionedCall’!dropout_2/StatefulPartitionedCall’!dropout_3/StatefulPartitionedCall’!dropout_4/StatefulPartitionedCall’!dropout_5/StatefulPartitionedCall’!dropout_6/StatefulPartitionedCall’!dropout_7/StatefulPartitionedCall’!dropout_8/StatefulPartitionedCall’+layer_normalization/StatefulPartitionedCall’-layer_normalization_1/StatefulPartitionedCall’-layer_normalization_2/StatefulPartitionedCall’-layer_normalization_3/StatefulPartitionedCall’-layer_normalization_4/StatefulPartitionedCall’-layer_normalization_5/StatefulPartitionedCall’-layer_normalization_6/StatefulPartitionedCall’-layer_normalization_7/StatefulPartitionedCall’,multi_head_attention/StatefulPartitionedCall’.multi_head_attention_1/StatefulPartitionedCall’.multi_head_attention_2/StatefulPartitionedCall’.multi_head_attention_3/StatefulPartitionedCall
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCallinput_1layer_normalization_15864layer_normalization_15866*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_13512»
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:04layer_normalization/StatefulPartitionedCall:output:0multi_head_attention_15869multi_head_attention_15871multi_head_attention_15873multi_head_attention_15875multi_head_attention_15877multi_head_attention_15879multi_head_attention_15881multi_head_attention_15883*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_15049ω
dropout/StatefulPartitionedCallStatefulPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_14978
tf.__operators__.add/AddV2AddV2(dropout/StatefulPartitionedCall:output:0input_1*
T0*,
_output_shapes
:?????????ΘΎ
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0layer_normalization_1_15888layer_normalization_1_15890*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_13601
conv1d/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0conv1d_15893conv1d_15895*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_13623
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_14935
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv1d_1_15899conv1d_1_15901*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_13651§
tf.__operators__.add_1/AddV2AddV2)conv1d_1/StatefulPartitionedCall:output:0tf.__operators__.add/AddV2:z:0*
T0*,
_output_shapes
:?????????Θΐ
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0layer_normalization_2_15905layer_normalization_2_15907*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_13680Σ
.multi_head_attention_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:06layer_normalization_2/StatefulPartitionedCall:output:0multi_head_attention_1_15910multi_head_attention_1_15912multi_head_attention_1_15914multi_head_attention_1_15916multi_head_attention_1_15918multi_head_attention_1_15920multi_head_attention_1_15922multi_head_attention_1_15924*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_14876£
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall7multi_head_attention_1/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_14805ͺ
tf.__operators__.add_2/AddV2AddV2*dropout_2/StatefulPartitionedCall:output:0 tf.__operators__.add_1/AddV2:z:0*
T0*,
_output_shapes
:?????????Θΐ
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0layer_normalization_3_15929layer_normalization_3_15931*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_13769’
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_3/StatefulPartitionedCall:output:0conv1d_2_15934conv1d_2_15936*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_13791
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_14762
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0conv1d_3_15940conv1d_3_15942*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_3_layer_call_and_return_conditional_losses_13819©
tf.__operators__.add_3/AddV2AddV2)conv1d_3/StatefulPartitionedCall:output:0 tf.__operators__.add_2/AddV2:z:0*
T0*,
_output_shapes
:?????????Θΐ
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_3/AddV2:z:0layer_normalization_4_15946layer_normalization_4_15948*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_13848Σ
.multi_head_attention_2/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:06layer_normalization_4/StatefulPartitionedCall:output:0multi_head_attention_2_15951multi_head_attention_2_15953multi_head_attention_2_15955multi_head_attention_2_15957multi_head_attention_2_15959multi_head_attention_2_15961multi_head_attention_2_15963multi_head_attention_2_15965*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_14703£
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall7multi_head_attention_2/StatefulPartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_14632ͺ
tf.__operators__.add_4/AddV2AddV2*dropout_4/StatefulPartitionedCall:output:0 tf.__operators__.add_3/AddV2:z:0*
T0*,
_output_shapes
:?????????Θΐ
-layer_normalization_5/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_4/AddV2:z:0layer_normalization_5_15970layer_normalization_5_15972*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_13937’
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_5/StatefulPartitionedCall:output:0conv1d_4_15975conv1d_4_15977*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_4_layer_call_and_return_conditional_losses_13959
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_14589
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0conv1d_5_15981conv1d_5_15983*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_5_layer_call_and_return_conditional_losses_13987©
tf.__operators__.add_5/AddV2AddV2)conv1d_5/StatefulPartitionedCall:output:0 tf.__operators__.add_4/AddV2:z:0*
T0*,
_output_shapes
:?????????Θΐ
-layer_normalization_6/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_5/AddV2:z:0layer_normalization_6_15987layer_normalization_6_15989*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_layer_normalization_6_layer_call_and_return_conditional_losses_14016Σ
.multi_head_attention_3/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_6/StatefulPartitionedCall:output:06layer_normalization_6/StatefulPartitionedCall:output:0multi_head_attention_3_15992multi_head_attention_3_15994multi_head_attention_3_15996multi_head_attention_3_15998multi_head_attention_3_16000multi_head_attention_3_16002multi_head_attention_3_16004multi_head_attention_3_16006*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_multi_head_attention_3_layer_call_and_return_conditional_losses_14530£
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall7multi_head_attention_3/StatefulPartitionedCall:output:0"^dropout_5/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_14459ͺ
tf.__operators__.add_6/AddV2AddV2*dropout_6/StatefulPartitionedCall:output:0 tf.__operators__.add_5/AddV2:z:0*
T0*,
_output_shapes
:?????????Θΐ
-layer_normalization_7/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_6/AddV2:z:0layer_normalization_7_16011layer_normalization_7_16013*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_layer_normalization_7_layer_call_and_return_conditional_losses_14105’
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_7/StatefulPartitionedCall:output:0conv1d_6_16016conv1d_6_16018*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_6_layer_call_and_return_conditional_losses_14127
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_14416
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0conv1d_7_16022conv1d_7_16024*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_7_layer_call_and_return_conditional_losses_14155©
tf.__operators__.add_7/AddV2AddV2)conv1d_7/StatefulPartitionedCall:output:0 tf.__operators__.add_6/AddV2:z:0*
T0*,
_output_shapes
:?????????Θς
(global_average_pooling1d/PartitionedCallPartitionedCall tf.__operators__.add_7/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_13480
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_16029dense_16031*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_14174
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_14373
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0dense_1_16035dense_1_16037*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_14197w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 

NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall.^layer_normalization_3/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall.^layer_normalization_5/StatefulPartitionedCall.^layer_normalization_6/StatefulPartitionedCall.^layer_normalization_7/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall/^multi_head_attention_1/StatefulPartitionedCall/^multi_head_attention_2/StatefulPartitionedCall/^multi_head_attention_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*΅
_input_shapes£
 :?????????Θ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2^
-layer_normalization_3/StatefulPartitionedCall-layer_normalization_3/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall2^
-layer_normalization_5/StatefulPartitionedCall-layer_normalization_5/StatefulPartitionedCall2^
-layer_normalization_6/StatefulPartitionedCall-layer_normalization_6/StatefulPartitionedCall2^
-layer_normalization_7/StatefulPartitionedCall-layer_normalization_7/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall2`
.multi_head_attention_1/StatefulPartitionedCall.multi_head_attention_1/StatefulPartitionedCall2`
.multi_head_attention_2/StatefulPartitionedCall.multi_head_attention_2/StatefulPartitionedCall2`
.multi_head_attention_3/StatefulPartitionedCall.multi_head_attention_3/StatefulPartitionedCall:U Q
,
_output_shapes
:?????????Θ
!
_user_specified_name	input_1


c
D__inference_dropout_7_layer_call_and_return_conditional_losses_14416

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:?????????ΘC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:?????????Θ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????Θt
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????Θn
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:?????????Θ^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:?????????Θ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
ύ

P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_17950

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(r
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:?????????Θ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¬
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θb
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:?????????Θ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θh
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θw
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????Θw
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Θg
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:?????????Θ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
·2
ϊ
Q__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_14876	
query	
valueA
+query_einsum_einsum_readvariableop_resource:23
!query_add_readvariableop_resource:2?
)key_einsum_einsum_readvariableop_resource:21
key_add_readvariableop_resource:2A
+value_einsum_einsum_readvariableop_resource:23
!value_add_readvariableop_resource:2L
6attention_output_einsum_einsum_readvariableop_resource:2:
,attention_output_add_readvariableop_resource:
identity’#attention_output/add/ReadVariableOp’-attention_output/einsum/Einsum/ReadVariableOp’key/add/ReadVariableOp’ key/einsum/Einsum/ReadVariableOp’query/add/ReadVariableOp’"query/einsum/Einsum/ReadVariableOp’value/add/ReadVariableOp’"value/einsum/Einsum/ReadVariableOp
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:2*
dtype0
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0­
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:2*
dtype0
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:2*
dtype0
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΓΠ>d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:?????????Θ2
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:?????????ΘΘ*
equationaecd,abcd->acbep
softmax_1/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:?????????ΘΘ]
dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ?
dropout_10/dropout/MulMulsoftmax_1/Softmax:softmax:0!dropout_10/dropout/Const:output:0*
T0*1
_output_shapes
:?????????ΘΘc
dropout_10/dropout/ShapeShapesoftmax_1/Softmax:softmax:0*
T0*
_output_shapes
:¬
/dropout_10/dropout/random_uniform/RandomUniformRandomUniform!dropout_10/dropout/Shape:output:0*
T0*1
_output_shapes
:?????????ΘΘ*
dtype0f
!dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >Ρ
dropout_10/dropout/GreaterEqualGreaterEqual8dropout_10/dropout/random_uniform/RandomUniform:output:0*dropout_10/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:?????????ΘΘ
dropout_10/dropout/CastCast#dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:?????????ΘΘ
dropout_10/dropout/Mul_1Muldropout_10/dropout/Mul:z:0dropout_10/dropout/Cast:y:0*
T0*1
_output_shapes
:?????????ΘΘ©
einsum_1/EinsumEinsumdropout_10/dropout/Mul_1:z:0value/add:z:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationacbe,aecd->abcd¨
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0Φ
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????Θ*
equationabcd,cde->abe
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0ͺ
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θl
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:?????????ΘΨ
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Θ:?????????Θ: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:?????????Θ

_user_specified_namequery:SO
,
_output_shapes
:?????????Θ

_user_specified_namevalue
κ

5__inference_layer_normalization_6_layer_call_fn_18214

inputs
unknown:
	unknown_0:
identity’StatefulPartitionedCallκ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_layer_normalization_6_layer_call_and_return_conditional_losses_14016t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
Τ

&__inference_conv1d_layer_call_fn_17566

inputs
unknown:
	unknown_0:
identity’StatefulPartitionedCallΫ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_13623t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
ύ

P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_13937

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(r
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:?????????Θ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¬
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θb
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:?????????Θ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θh
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θw
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????Θw
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Θg
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:?????????Θ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
Ί
―C
@__inference_model_layer_call_and_return_conditional_losses_17347

inputsG
9layer_normalization_batchnorm_mul_readvariableop_resource:C
5layer_normalization_batchnorm_readvariableop_resource:V
@multi_head_attention_query_einsum_einsum_readvariableop_resource:2H
6multi_head_attention_query_add_readvariableop_resource:2T
>multi_head_attention_key_einsum_einsum_readvariableop_resource:2F
4multi_head_attention_key_add_readvariableop_resource:2V
@multi_head_attention_value_einsum_einsum_readvariableop_resource:2H
6multi_head_attention_value_add_readvariableop_resource:2a
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource:2O
Amulti_head_attention_attention_output_add_readvariableop_resource:I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:E
7layer_normalization_1_batchnorm_readvariableop_resource:H
2conv1d_conv1d_expanddims_1_readvariableop_resource:4
&conv1d_biasadd_readvariableop_resource:J
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_1_biasadd_readvariableop_resource:I
;layer_normalization_2_batchnorm_mul_readvariableop_resource:E
7layer_normalization_2_batchnorm_readvariableop_resource:X
Bmulti_head_attention_1_query_einsum_einsum_readvariableop_resource:2J
8multi_head_attention_1_query_add_readvariableop_resource:2V
@multi_head_attention_1_key_einsum_einsum_readvariableop_resource:2H
6multi_head_attention_1_key_add_readvariableop_resource:2X
Bmulti_head_attention_1_value_einsum_einsum_readvariableop_resource:2J
8multi_head_attention_1_value_add_readvariableop_resource:2c
Mmulti_head_attention_1_attention_output_einsum_einsum_readvariableop_resource:2Q
Cmulti_head_attention_1_attention_output_add_readvariableop_resource:I
;layer_normalization_3_batchnorm_mul_readvariableop_resource:E
7layer_normalization_3_batchnorm_readvariableop_resource:J
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_2_biasadd_readvariableop_resource:J
4conv1d_3_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_3_biasadd_readvariableop_resource:I
;layer_normalization_4_batchnorm_mul_readvariableop_resource:E
7layer_normalization_4_batchnorm_readvariableop_resource:X
Bmulti_head_attention_2_query_einsum_einsum_readvariableop_resource:2J
8multi_head_attention_2_query_add_readvariableop_resource:2V
@multi_head_attention_2_key_einsum_einsum_readvariableop_resource:2H
6multi_head_attention_2_key_add_readvariableop_resource:2X
Bmulti_head_attention_2_value_einsum_einsum_readvariableop_resource:2J
8multi_head_attention_2_value_add_readvariableop_resource:2c
Mmulti_head_attention_2_attention_output_einsum_einsum_readvariableop_resource:2Q
Cmulti_head_attention_2_attention_output_add_readvariableop_resource:I
;layer_normalization_5_batchnorm_mul_readvariableop_resource:E
7layer_normalization_5_batchnorm_readvariableop_resource:J
4conv1d_4_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_4_biasadd_readvariableop_resource:J
4conv1d_5_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_5_biasadd_readvariableop_resource:I
;layer_normalization_6_batchnorm_mul_readvariableop_resource:E
7layer_normalization_6_batchnorm_readvariableop_resource:X
Bmulti_head_attention_3_query_einsum_einsum_readvariableop_resource:2J
8multi_head_attention_3_query_add_readvariableop_resource:2V
@multi_head_attention_3_key_einsum_einsum_readvariableop_resource:2H
6multi_head_attention_3_key_add_readvariableop_resource:2X
Bmulti_head_attention_3_value_einsum_einsum_readvariableop_resource:2J
8multi_head_attention_3_value_add_readvariableop_resource:2c
Mmulti_head_attention_3_attention_output_einsum_einsum_readvariableop_resource:2Q
Cmulti_head_attention_3_attention_output_add_readvariableop_resource:I
;layer_normalization_7_batchnorm_mul_readvariableop_resource:E
7layer_normalization_7_batchnorm_readvariableop_resource:J
4conv1d_6_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_6_biasadd_readvariableop_resource:J
4conv1d_7_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_7_biasadd_readvariableop_resource:8
$dense_matmul_readvariableop_resource:
Θ4
%dense_biasadd_readvariableop_resource:	9
&dense_1_matmul_readvariableop_resource:	5
'dense_1_biasadd_readvariableop_resource:
identity’conv1d/BiasAdd/ReadVariableOp’)conv1d/Conv1D/ExpandDims_1/ReadVariableOp’conv1d_1/BiasAdd/ReadVariableOp’+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp’conv1d_2/BiasAdd/ReadVariableOp’+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp’conv1d_3/BiasAdd/ReadVariableOp’+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp’conv1d_4/BiasAdd/ReadVariableOp’+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp’conv1d_5/BiasAdd/ReadVariableOp’+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp’conv1d_6/BiasAdd/ReadVariableOp’+conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp’conv1d_7/BiasAdd/ReadVariableOp’+conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp’dense/BiasAdd/ReadVariableOp’dense/MatMul/ReadVariableOp’dense_1/BiasAdd/ReadVariableOp’dense_1/MatMul/ReadVariableOp’,layer_normalization/batchnorm/ReadVariableOp’0layer_normalization/batchnorm/mul/ReadVariableOp’.layer_normalization_1/batchnorm/ReadVariableOp’2layer_normalization_1/batchnorm/mul/ReadVariableOp’.layer_normalization_2/batchnorm/ReadVariableOp’2layer_normalization_2/batchnorm/mul/ReadVariableOp’.layer_normalization_3/batchnorm/ReadVariableOp’2layer_normalization_3/batchnorm/mul/ReadVariableOp’.layer_normalization_4/batchnorm/ReadVariableOp’2layer_normalization_4/batchnorm/mul/ReadVariableOp’.layer_normalization_5/batchnorm/ReadVariableOp’2layer_normalization_5/batchnorm/mul/ReadVariableOp’.layer_normalization_6/batchnorm/ReadVariableOp’2layer_normalization_6/batchnorm/mul/ReadVariableOp’.layer_normalization_7/batchnorm/ReadVariableOp’2layer_normalization_7/batchnorm/mul/ReadVariableOp’8multi_head_attention/attention_output/add/ReadVariableOp’Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp’+multi_head_attention/key/add/ReadVariableOp’5multi_head_attention/key/einsum/Einsum/ReadVariableOp’-multi_head_attention/query/add/ReadVariableOp’7multi_head_attention/query/einsum/Einsum/ReadVariableOp’-multi_head_attention/value/add/ReadVariableOp’7multi_head_attention/value/einsum/Einsum/ReadVariableOp’:multi_head_attention_1/attention_output/add/ReadVariableOp’Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp’-multi_head_attention_1/key/add/ReadVariableOp’7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp’/multi_head_attention_1/query/add/ReadVariableOp’9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp’/multi_head_attention_1/value/add/ReadVariableOp’9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp’:multi_head_attention_2/attention_output/add/ReadVariableOp’Dmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp’-multi_head_attention_2/key/add/ReadVariableOp’7multi_head_attention_2/key/einsum/Einsum/ReadVariableOp’/multi_head_attention_2/query/add/ReadVariableOp’9multi_head_attention_2/query/einsum/Einsum/ReadVariableOp’/multi_head_attention_2/value/add/ReadVariableOp’9multi_head_attention_2/value/einsum/Einsum/ReadVariableOp’:multi_head_attention_3/attention_output/add/ReadVariableOp’Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp’-multi_head_attention_3/key/add/ReadVariableOp’7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp’/multi_head_attention_3/query/add/ReadVariableOp’9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp’/multi_head_attention_3/value/add/ReadVariableOp’9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:΅
 layer_normalization/moments/meanMeaninputs;layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:?????????Θ΄
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceinputs1layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θ
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:θ
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Ύ
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θ
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:?????????Θ¦
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Β
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ
#layer_normalization/batchnorm/mul_1Mulinputs%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θ³
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θ
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Ύ
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????Θ³
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????ΘΌ
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0ύ
(multi_head_attention/query/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde€
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:2*
dtype0Μ
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2Έ
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0ω
&multi_head_attention/key/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde 
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:2*
dtype0Ζ
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2Ό
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0ύ
(multi_head_attention/value/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde€
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:2*
dtype0Μ
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2_
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΓΠ>£
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:?????????Θ2Π
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*1
_output_shapes
:?????????ΘΘ*
equationaecd,abcd->acbe
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*1
_output_shapes
:?????????ΘΘq
,multi_head_attention/dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ?Τ
*multi_head_attention/dropout_9/dropout/MulMul.multi_head_attention/softmax/Softmax:softmax:05multi_head_attention/dropout_9/dropout/Const:output:0*
T0*1
_output_shapes
:?????????ΘΘ
,multi_head_attention/dropout_9/dropout/ShapeShape.multi_head_attention/softmax/Softmax:softmax:0*
T0*
_output_shapes
:Τ
Cmulti_head_attention/dropout_9/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention/dropout_9/dropout/Shape:output:0*
T0*1
_output_shapes
:?????????ΘΘ*
dtype0z
5multi_head_attention/dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >
3multi_head_attention/dropout_9/dropout/GreaterEqualGreaterEqualLmulti_head_attention/dropout_9/dropout/random_uniform/RandomUniform:output:0>multi_head_attention/dropout_9/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:?????????ΘΘ·
+multi_head_attention/dropout_9/dropout/CastCast7multi_head_attention/dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:?????????ΘΘΠ
,multi_head_attention/dropout_9/dropout/Mul_1Mul.multi_head_attention/dropout_9/dropout/Mul:z:0/multi_head_attention/dropout_9/dropout/Cast:y:0*
T0*1
_output_shapes
:?????????ΘΘη
$multi_head_attention/einsum_1/EinsumEinsum0multi_head_attention/dropout_9/dropout/Mul_1:z:0"multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationacbe,aecd->abcd?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????Θ*
equationabcd,cde->abeΆ
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0ι
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????ΘZ
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ? 
dropout/dropout/MulMul-multi_head_attention/attention_output/add:z:0dropout/dropout/Const:output:0*
T0*,
_output_shapes
:?????????Θr
dropout/dropout/ShapeShape-multi_head_attention/attention_output/add:z:0*
T0*
_output_shapes
:‘
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*,
_output_shapes
:?????????Θ*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >Γ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????Θ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????Θ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*,
_output_shapes
:?????????Θ}
tf.__operators__.add/AddV2AddV2dropout/dropout/Mul_1:z:0inputs*
T0*,
_output_shapes
:?????????Θ~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Ρ
"layer_normalization_1/moments/meanMeantf.__operators__.add/AddV2:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:?????????ΘΠ
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferencetf.__operators__.add/AddV2:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θ
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ξ
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Δ
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θ
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:?????????Θͺ
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Θ
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ¬
%layer_normalization_1/batchnorm/mul_1Multf.__operators__.add/AddV2:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ΘΉ
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θ’
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Δ
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????ΘΉ
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Θg
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????³
conv1d/Conv1D/ExpandDims
ExpandDims)layer_normalization_1/batchnorm/add_1:z:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ 
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ΅
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Γ
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θc
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:?????????Θ\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ?
dropout_1/dropout/MulMulconv1d/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*,
_output_shapes
:?????????Θ`
dropout_1/dropout/ShapeShapeconv1d/Relu:activations:0*
T0*
_output_shapes
:₯
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*,
_output_shapes
:?????????Θ*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >Ι
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????Θ
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????Θ
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*,
_output_shapes
:?????????Θi
conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????©
conv1d_1/Conv1D/ExpandDims
ExpandDimsdropout_1/dropout/Mul_1:z:0'conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ€
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_1/Conv1D/ExpandDims_1
ExpandDims3conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Ι
conv1d_1/Conv1DConv2D#conv1d_1/Conv1D/ExpandDims:output:0%conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

conv1d_1/Conv1D/SqueezeSqueezeconv1d_1/Conv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv1d_1/BiasAddBiasAdd conv1d_1/Conv1D/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ
tf.__operators__.add_1/AddV2AddV2conv1d_1/BiasAdd:output:0tf.__operators__.add/AddV2:z:0*
T0*,
_output_shapes
:?????????Θ~
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Σ
"layer_normalization_2/moments/meanMean tf.__operators__.add_1/AddV2:z:0=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*,
_output_shapes
:?????????Θ?
/layer_normalization_2/moments/SquaredDifferenceSquaredDifference tf.__operators__.add_1/AddV2:z:03layer_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θ
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ξ
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(j
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Δ
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θ
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*,
_output_shapes
:?????????Θͺ
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Θ
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ?
%layer_normalization_2/batchnorm/mul_1Mul tf.__operators__.add_1/AddV2:z:0'layer_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ΘΉ
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θ’
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Δ
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????ΘΉ
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Θΐ
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0
*multi_head_attention_1/query/einsum/EinsumEinsum)layer_normalization_2/batchnorm/add_1:z:0Amulti_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde¨
/multi_head_attention_1/query/add/ReadVariableOpReadVariableOp8multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

:2*
dtype0?
 multi_head_attention_1/query/addAddV23multi_head_attention_1/query/einsum/Einsum:output:07multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2Ό
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0?
(multi_head_attention_1/key/einsum/EinsumEinsum)layer_normalization_2/batchnorm/add_1:z:0?multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde€
-multi_head_attention_1/key/add/ReadVariableOpReadVariableOp6multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

:2*
dtype0Μ
multi_head_attention_1/key/addAddV21multi_head_attention_1/key/einsum/Einsum:output:05multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2ΐ
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0
*multi_head_attention_1/value/einsum/EinsumEinsum)layer_normalization_2/batchnorm/add_1:z:0Amulti_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde¨
/multi_head_attention_1/value/add/ReadVariableOpReadVariableOp8multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

:2*
dtype0?
 multi_head_attention_1/value/addAddV23multi_head_attention_1/value/einsum/Einsum:output:07multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2a
multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΓΠ>©
multi_head_attention_1/MulMul$multi_head_attention_1/query/add:z:0%multi_head_attention_1/Mul/y:output:0*
T0*0
_output_shapes
:?????????Θ2Φ
$multi_head_attention_1/einsum/EinsumEinsum"multi_head_attention_1/key/add:z:0multi_head_attention_1/Mul:z:0*
N*
T0*1
_output_shapes
:?????????ΘΘ*
equationaecd,abcd->acbe
(multi_head_attention_1/softmax_1/SoftmaxSoftmax-multi_head_attention_1/einsum/Einsum:output:0*
T0*1
_output_shapes
:?????????ΘΘt
/multi_head_attention_1/dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ?ή
-multi_head_attention_1/dropout_10/dropout/MulMul2multi_head_attention_1/softmax_1/Softmax:softmax:08multi_head_attention_1/dropout_10/dropout/Const:output:0*
T0*1
_output_shapes
:?????????ΘΘ
/multi_head_attention_1/dropout_10/dropout/ShapeShape2multi_head_attention_1/softmax_1/Softmax:softmax:0*
T0*
_output_shapes
:Ϊ
Fmulti_head_attention_1/dropout_10/dropout/random_uniform/RandomUniformRandomUniform8multi_head_attention_1/dropout_10/dropout/Shape:output:0*
T0*1
_output_shapes
:?????????ΘΘ*
dtype0}
8multi_head_attention_1/dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >
6multi_head_attention_1/dropout_10/dropout/GreaterEqualGreaterEqualOmulti_head_attention_1/dropout_10/dropout/random_uniform/RandomUniform:output:0Amulti_head_attention_1/dropout_10/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:?????????ΘΘ½
.multi_head_attention_1/dropout_10/dropout/CastCast:multi_head_attention_1/dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:?????????ΘΘΩ
/multi_head_attention_1/dropout_10/dropout/Mul_1Mul1multi_head_attention_1/dropout_10/dropout/Mul:z:02multi_head_attention_1/dropout_10/dropout/Cast:y:0*
T0*1
_output_shapes
:?????????ΘΘξ
&multi_head_attention_1/einsum_1/EinsumEinsum3multi_head_attention_1/dropout_10/dropout/Mul_1:z:0$multi_head_attention_1/value/add:z:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationacbe,aecd->abcdΦ
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0
5multi_head_attention_1/attention_output/einsum/EinsumEinsum/multi_head_attention_1/einsum_1/Einsum:output:0Lmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????Θ*
equationabcd,cde->abeΊ
:multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0ο
+multi_head_attention_1/attention_output/addAddV2>multi_head_attention_1/attention_output/einsum/Einsum:output:0Bmulti_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ?¦
dropout_2/dropout/MulMul/multi_head_attention_1/attention_output/add:z:0 dropout_2/dropout/Const:output:0*
T0*,
_output_shapes
:?????????Θv
dropout_2/dropout/ShapeShape/multi_head_attention_1/attention_output/add:z:0*
T0*
_output_shapes
:₯
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*,
_output_shapes
:?????????Θ*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >Ι
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????Θ
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????Θ
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*,
_output_shapes
:?????????Θ
tf.__operators__.add_2/AddV2AddV2dropout_2/dropout/Mul_1:z:0 tf.__operators__.add_1/AddV2:z:0*
T0*,
_output_shapes
:?????????Θ~
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Σ
"layer_normalization_3/moments/meanMean tf.__operators__.add_2/AddV2:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*,
_output_shapes
:?????????Θ?
/layer_normalization_3/moments/SquaredDifferenceSquaredDifference tf.__operators__.add_2/AddV2:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θ
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ξ
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(j
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Δ
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θ
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*,
_output_shapes
:?????????Θͺ
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Θ
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ?
%layer_normalization_3/batchnorm/mul_1Mul tf.__operators__.add_2/AddV2:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ΘΉ
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θ’
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Δ
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????ΘΉ
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Θi
conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????·
conv1d_2/Conv1D/ExpandDims
ExpandDims)layer_normalization_3/batchnorm/add_1:z:0'conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ€
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_2/Conv1D/ExpandDims_1
ExpandDims3conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Ι
conv1d_2/Conv1DConv2D#conv1d_2/Conv1D/ExpandDims:output:0%conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

conv1d_2/Conv1D/SqueezeSqueezeconv1d_2/Conv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv1d_2/BiasAddBiasAdd conv1d_2/Conv1D/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θg
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*,
_output_shapes
:?????????Θ\
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ?
dropout_3/dropout/MulMulconv1d_2/Relu:activations:0 dropout_3/dropout/Const:output:0*
T0*,
_output_shapes
:?????????Θb
dropout_3/dropout/ShapeShapeconv1d_2/Relu:activations:0*
T0*
_output_shapes
:₯
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*,
_output_shapes
:?????????Θ*
dtype0e
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >Ι
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????Θ
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????Θ
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*,
_output_shapes
:?????????Θi
conv1d_3/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????©
conv1d_3/Conv1D/ExpandDims
ExpandDimsdropout_3/dropout/Mul_1:z:0'conv1d_3/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ€
+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_3/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_3/Conv1D/ExpandDims_1
ExpandDims3conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Ι
conv1d_3/Conv1DConv2D#conv1d_3/Conv1D/ExpandDims:output:0%conv1d_3/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

conv1d_3/Conv1D/SqueezeSqueezeconv1d_3/Conv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv1d_3/BiasAddBiasAdd conv1d_3/Conv1D/Squeeze:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ
tf.__operators__.add_3/AddV2AddV2conv1d_3/BiasAdd:output:0 tf.__operators__.add_2/AddV2:z:0*
T0*,
_output_shapes
:?????????Θ~
4layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Σ
"layer_normalization_4/moments/meanMean tf.__operators__.add_3/AddV2:z:0=layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(
*layer_normalization_4/moments/StopGradientStopGradient+layer_normalization_4/moments/mean:output:0*
T0*,
_output_shapes
:?????????Θ?
/layer_normalization_4/moments/SquaredDifferenceSquaredDifference tf.__operators__.add_3/AddV2:z:03layer_normalization_4/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θ
8layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ξ
&layer_normalization_4/moments/varianceMean3layer_normalization_4/moments/SquaredDifference:z:0Alayer_normalization_4/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(j
%layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Δ
#layer_normalization_4/batchnorm/addAddV2/layer_normalization_4/moments/variance:output:0.layer_normalization_4/batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θ
%layer_normalization_4/batchnorm/RsqrtRsqrt'layer_normalization_4/batchnorm/add:z:0*
T0*,
_output_shapes
:?????????Θͺ
2layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Θ
#layer_normalization_4/batchnorm/mulMul)layer_normalization_4/batchnorm/Rsqrt:y:0:layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ?
%layer_normalization_4/batchnorm/mul_1Mul tf.__operators__.add_3/AddV2:z:0'layer_normalization_4/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ΘΉ
%layer_normalization_4/batchnorm/mul_2Mul+layer_normalization_4/moments/mean:output:0'layer_normalization_4/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θ’
.layer_normalization_4/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Δ
#layer_normalization_4/batchnorm/subSub6layer_normalization_4/batchnorm/ReadVariableOp:value:0)layer_normalization_4/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????ΘΉ
%layer_normalization_4/batchnorm/add_1AddV2)layer_normalization_4/batchnorm/mul_1:z:0'layer_normalization_4/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Θΐ
9multi_head_attention_2/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_2_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0
*multi_head_attention_2/query/einsum/EinsumEinsum)layer_normalization_4/batchnorm/add_1:z:0Amulti_head_attention_2/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde¨
/multi_head_attention_2/query/add/ReadVariableOpReadVariableOp8multi_head_attention_2_query_add_readvariableop_resource*
_output_shapes

:2*
dtype0?
 multi_head_attention_2/query/addAddV23multi_head_attention_2/query/einsum/Einsum:output:07multi_head_attention_2/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2Ό
7multi_head_attention_2/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_2_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0?
(multi_head_attention_2/key/einsum/EinsumEinsum)layer_normalization_4/batchnorm/add_1:z:0?multi_head_attention_2/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde€
-multi_head_attention_2/key/add/ReadVariableOpReadVariableOp6multi_head_attention_2_key_add_readvariableop_resource*
_output_shapes

:2*
dtype0Μ
multi_head_attention_2/key/addAddV21multi_head_attention_2/key/einsum/Einsum:output:05multi_head_attention_2/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2ΐ
9multi_head_attention_2/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_2_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0
*multi_head_attention_2/value/einsum/EinsumEinsum)layer_normalization_4/batchnorm/add_1:z:0Amulti_head_attention_2/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde¨
/multi_head_attention_2/value/add/ReadVariableOpReadVariableOp8multi_head_attention_2_value_add_readvariableop_resource*
_output_shapes

:2*
dtype0?
 multi_head_attention_2/value/addAddV23multi_head_attention_2/value/einsum/Einsum:output:07multi_head_attention_2/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2a
multi_head_attention_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΓΠ>©
multi_head_attention_2/MulMul$multi_head_attention_2/query/add:z:0%multi_head_attention_2/Mul/y:output:0*
T0*0
_output_shapes
:?????????Θ2Φ
$multi_head_attention_2/einsum/EinsumEinsum"multi_head_attention_2/key/add:z:0multi_head_attention_2/Mul:z:0*
N*
T0*1
_output_shapes
:?????????ΘΘ*
equationaecd,abcd->acbe
(multi_head_attention_2/softmax_2/SoftmaxSoftmax-multi_head_attention_2/einsum/Einsum:output:0*
T0*1
_output_shapes
:?????????ΘΘt
/multi_head_attention_2/dropout_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ?ή
-multi_head_attention_2/dropout_11/dropout/MulMul2multi_head_attention_2/softmax_2/Softmax:softmax:08multi_head_attention_2/dropout_11/dropout/Const:output:0*
T0*1
_output_shapes
:?????????ΘΘ
/multi_head_attention_2/dropout_11/dropout/ShapeShape2multi_head_attention_2/softmax_2/Softmax:softmax:0*
T0*
_output_shapes
:Ϊ
Fmulti_head_attention_2/dropout_11/dropout/random_uniform/RandomUniformRandomUniform8multi_head_attention_2/dropout_11/dropout/Shape:output:0*
T0*1
_output_shapes
:?????????ΘΘ*
dtype0}
8multi_head_attention_2/dropout_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >
6multi_head_attention_2/dropout_11/dropout/GreaterEqualGreaterEqualOmulti_head_attention_2/dropout_11/dropout/random_uniform/RandomUniform:output:0Amulti_head_attention_2/dropout_11/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:?????????ΘΘ½
.multi_head_attention_2/dropout_11/dropout/CastCast:multi_head_attention_2/dropout_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:?????????ΘΘΩ
/multi_head_attention_2/dropout_11/dropout/Mul_1Mul1multi_head_attention_2/dropout_11/dropout/Mul:z:02multi_head_attention_2/dropout_11/dropout/Cast:y:0*
T0*1
_output_shapes
:?????????ΘΘξ
&multi_head_attention_2/einsum_1/EinsumEinsum3multi_head_attention_2/dropout_11/dropout/Mul_1:z:0$multi_head_attention_2/value/add:z:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationacbe,aecd->abcdΦ
Dmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_2_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0
5multi_head_attention_2/attention_output/einsum/EinsumEinsum/multi_head_attention_2/einsum_1/Einsum:output:0Lmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????Θ*
equationabcd,cde->abeΊ
:multi_head_attention_2/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_2_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0ο
+multi_head_attention_2/attention_output/addAddV2>multi_head_attention_2/attention_output/einsum/Einsum:output:0Bmulti_head_attention_2/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ\
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ?¦
dropout_4/dropout/MulMul/multi_head_attention_2/attention_output/add:z:0 dropout_4/dropout/Const:output:0*
T0*,
_output_shapes
:?????????Θv
dropout_4/dropout/ShapeShape/multi_head_attention_2/attention_output/add:z:0*
T0*
_output_shapes
:₯
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*,
_output_shapes
:?????????Θ*
dtype0e
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >Ι
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????Θ
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????Θ
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*,
_output_shapes
:?????????Θ
tf.__operators__.add_4/AddV2AddV2dropout_4/dropout/Mul_1:z:0 tf.__operators__.add_3/AddV2:z:0*
T0*,
_output_shapes
:?????????Θ~
4layer_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Σ
"layer_normalization_5/moments/meanMean tf.__operators__.add_4/AddV2:z:0=layer_normalization_5/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(
*layer_normalization_5/moments/StopGradientStopGradient+layer_normalization_5/moments/mean:output:0*
T0*,
_output_shapes
:?????????Θ?
/layer_normalization_5/moments/SquaredDifferenceSquaredDifference tf.__operators__.add_4/AddV2:z:03layer_normalization_5/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θ
8layer_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ξ
&layer_normalization_5/moments/varianceMean3layer_normalization_5/moments/SquaredDifference:z:0Alayer_normalization_5/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(j
%layer_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Δ
#layer_normalization_5/batchnorm/addAddV2/layer_normalization_5/moments/variance:output:0.layer_normalization_5/batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θ
%layer_normalization_5/batchnorm/RsqrtRsqrt'layer_normalization_5/batchnorm/add:z:0*
T0*,
_output_shapes
:?????????Θͺ
2layer_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Θ
#layer_normalization_5/batchnorm/mulMul)layer_normalization_5/batchnorm/Rsqrt:y:0:layer_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ?
%layer_normalization_5/batchnorm/mul_1Mul tf.__operators__.add_4/AddV2:z:0'layer_normalization_5/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ΘΉ
%layer_normalization_5/batchnorm/mul_2Mul+layer_normalization_5/moments/mean:output:0'layer_normalization_5/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θ’
.layer_normalization_5/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Δ
#layer_normalization_5/batchnorm/subSub6layer_normalization_5/batchnorm/ReadVariableOp:value:0)layer_normalization_5/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????ΘΉ
%layer_normalization_5/batchnorm/add_1AddV2)layer_normalization_5/batchnorm/mul_1:z:0'layer_normalization_5/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Θi
conv1d_4/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????·
conv1d_4/Conv1D/ExpandDims
ExpandDims)layer_normalization_5/batchnorm/add_1:z:0'conv1d_4/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ€
+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_4/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_4/Conv1D/ExpandDims_1
ExpandDims3conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_4/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Ι
conv1d_4/Conv1DConv2D#conv1d_4/Conv1D/ExpandDims:output:0%conv1d_4/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

conv1d_4/Conv1D/SqueezeSqueezeconv1d_4/Conv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????
conv1d_4/BiasAdd/ReadVariableOpReadVariableOp(conv1d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv1d_4/BiasAddBiasAdd conv1d_4/Conv1D/Squeeze:output:0'conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θg
conv1d_4/ReluReluconv1d_4/BiasAdd:output:0*
T0*,
_output_shapes
:?????????Θ\
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ?
dropout_5/dropout/MulMulconv1d_4/Relu:activations:0 dropout_5/dropout/Const:output:0*
T0*,
_output_shapes
:?????????Θb
dropout_5/dropout/ShapeShapeconv1d_4/Relu:activations:0*
T0*
_output_shapes
:₯
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*,
_output_shapes
:?????????Θ*
dtype0e
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >Ι
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????Θ
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????Θ
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*,
_output_shapes
:?????????Θi
conv1d_5/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????©
conv1d_5/Conv1D/ExpandDims
ExpandDimsdropout_5/dropout/Mul_1:z:0'conv1d_5/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ€
+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_5/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_5/Conv1D/ExpandDims_1
ExpandDims3conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_5/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Ι
conv1d_5/Conv1DConv2D#conv1d_5/Conv1D/ExpandDims:output:0%conv1d_5/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

conv1d_5/Conv1D/SqueezeSqueezeconv1d_5/Conv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????
conv1d_5/BiasAdd/ReadVariableOpReadVariableOp(conv1d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv1d_5/BiasAddBiasAdd conv1d_5/Conv1D/Squeeze:output:0'conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ
tf.__operators__.add_5/AddV2AddV2conv1d_5/BiasAdd:output:0 tf.__operators__.add_4/AddV2:z:0*
T0*,
_output_shapes
:?????????Θ~
4layer_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Σ
"layer_normalization_6/moments/meanMean tf.__operators__.add_5/AddV2:z:0=layer_normalization_6/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(
*layer_normalization_6/moments/StopGradientStopGradient+layer_normalization_6/moments/mean:output:0*
T0*,
_output_shapes
:?????????Θ?
/layer_normalization_6/moments/SquaredDifferenceSquaredDifference tf.__operators__.add_5/AddV2:z:03layer_normalization_6/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θ
8layer_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ξ
&layer_normalization_6/moments/varianceMean3layer_normalization_6/moments/SquaredDifference:z:0Alayer_normalization_6/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(j
%layer_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Δ
#layer_normalization_6/batchnorm/addAddV2/layer_normalization_6/moments/variance:output:0.layer_normalization_6/batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θ
%layer_normalization_6/batchnorm/RsqrtRsqrt'layer_normalization_6/batchnorm/add:z:0*
T0*,
_output_shapes
:?????????Θͺ
2layer_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Θ
#layer_normalization_6/batchnorm/mulMul)layer_normalization_6/batchnorm/Rsqrt:y:0:layer_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ?
%layer_normalization_6/batchnorm/mul_1Mul tf.__operators__.add_5/AddV2:z:0'layer_normalization_6/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ΘΉ
%layer_normalization_6/batchnorm/mul_2Mul+layer_normalization_6/moments/mean:output:0'layer_normalization_6/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θ’
.layer_normalization_6/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Δ
#layer_normalization_6/batchnorm/subSub6layer_normalization_6/batchnorm/ReadVariableOp:value:0)layer_normalization_6/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????ΘΉ
%layer_normalization_6/batchnorm/add_1AddV2)layer_normalization_6/batchnorm/mul_1:z:0'layer_normalization_6/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Θΐ
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_3_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0
*multi_head_attention_3/query/einsum/EinsumEinsum)layer_normalization_6/batchnorm/add_1:z:0Amulti_head_attention_3/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde¨
/multi_head_attention_3/query/add/ReadVariableOpReadVariableOp8multi_head_attention_3_query_add_readvariableop_resource*
_output_shapes

:2*
dtype0?
 multi_head_attention_3/query/addAddV23multi_head_attention_3/query/einsum/Einsum:output:07multi_head_attention_3/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2Ό
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_3_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0?
(multi_head_attention_3/key/einsum/EinsumEinsum)layer_normalization_6/batchnorm/add_1:z:0?multi_head_attention_3/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde€
-multi_head_attention_3/key/add/ReadVariableOpReadVariableOp6multi_head_attention_3_key_add_readvariableop_resource*
_output_shapes

:2*
dtype0Μ
multi_head_attention_3/key/addAddV21multi_head_attention_3/key/einsum/Einsum:output:05multi_head_attention_3/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2ΐ
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_3_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0
*multi_head_attention_3/value/einsum/EinsumEinsum)layer_normalization_6/batchnorm/add_1:z:0Amulti_head_attention_3/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abde¨
/multi_head_attention_3/value/add/ReadVariableOpReadVariableOp8multi_head_attention_3_value_add_readvariableop_resource*
_output_shapes

:2*
dtype0?
 multi_head_attention_3/value/addAddV23multi_head_attention_3/value/einsum/Einsum:output:07multi_head_attention_3/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2a
multi_head_attention_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΓΠ>©
multi_head_attention_3/MulMul$multi_head_attention_3/query/add:z:0%multi_head_attention_3/Mul/y:output:0*
T0*0
_output_shapes
:?????????Θ2Φ
$multi_head_attention_3/einsum/EinsumEinsum"multi_head_attention_3/key/add:z:0multi_head_attention_3/Mul:z:0*
N*
T0*1
_output_shapes
:?????????ΘΘ*
equationaecd,abcd->acbe
(multi_head_attention_3/softmax_3/SoftmaxSoftmax-multi_head_attention_3/einsum/Einsum:output:0*
T0*1
_output_shapes
:?????????ΘΘt
/multi_head_attention_3/dropout_12/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ?ή
-multi_head_attention_3/dropout_12/dropout/MulMul2multi_head_attention_3/softmax_3/Softmax:softmax:08multi_head_attention_3/dropout_12/dropout/Const:output:0*
T0*1
_output_shapes
:?????????ΘΘ
/multi_head_attention_3/dropout_12/dropout/ShapeShape2multi_head_attention_3/softmax_3/Softmax:softmax:0*
T0*
_output_shapes
:Ϊ
Fmulti_head_attention_3/dropout_12/dropout/random_uniform/RandomUniformRandomUniform8multi_head_attention_3/dropout_12/dropout/Shape:output:0*
T0*1
_output_shapes
:?????????ΘΘ*
dtype0}
8multi_head_attention_3/dropout_12/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >
6multi_head_attention_3/dropout_12/dropout/GreaterEqualGreaterEqualOmulti_head_attention_3/dropout_12/dropout/random_uniform/RandomUniform:output:0Amulti_head_attention_3/dropout_12/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:?????????ΘΘ½
.multi_head_attention_3/dropout_12/dropout/CastCast:multi_head_attention_3/dropout_12/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:?????????ΘΘΩ
/multi_head_attention_3/dropout_12/dropout/Mul_1Mul1multi_head_attention_3/dropout_12/dropout/Mul:z:02multi_head_attention_3/dropout_12/dropout/Cast:y:0*
T0*1
_output_shapes
:?????????ΘΘξ
&multi_head_attention_3/einsum_1/EinsumEinsum3multi_head_attention_3/dropout_12/dropout/Mul_1:z:0$multi_head_attention_3/value/add:z:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationacbe,aecd->abcdΦ
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_3_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0
5multi_head_attention_3/attention_output/einsum/EinsumEinsum/multi_head_attention_3/einsum_1/Einsum:output:0Lmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????Θ*
equationabcd,cde->abeΊ
:multi_head_attention_3/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_3_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0ο
+multi_head_attention_3/attention_output/addAddV2>multi_head_attention_3/attention_output/einsum/Einsum:output:0Bmulti_head_attention_3/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ\
dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ?¦
dropout_6/dropout/MulMul/multi_head_attention_3/attention_output/add:z:0 dropout_6/dropout/Const:output:0*
T0*,
_output_shapes
:?????????Θv
dropout_6/dropout/ShapeShape/multi_head_attention_3/attention_output/add:z:0*
T0*
_output_shapes
:₯
.dropout_6/dropout/random_uniform/RandomUniformRandomUniform dropout_6/dropout/Shape:output:0*
T0*,
_output_shapes
:?????????Θ*
dtype0e
 dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >Ι
dropout_6/dropout/GreaterEqualGreaterEqual7dropout_6/dropout/random_uniform/RandomUniform:output:0)dropout_6/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????Θ
dropout_6/dropout/CastCast"dropout_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????Θ
dropout_6/dropout/Mul_1Muldropout_6/dropout/Mul:z:0dropout_6/dropout/Cast:y:0*
T0*,
_output_shapes
:?????????Θ
tf.__operators__.add_6/AddV2AddV2dropout_6/dropout/Mul_1:z:0 tf.__operators__.add_5/AddV2:z:0*
T0*,
_output_shapes
:?????????Θ~
4layer_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Σ
"layer_normalization_7/moments/meanMean tf.__operators__.add_6/AddV2:z:0=layer_normalization_7/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(
*layer_normalization_7/moments/StopGradientStopGradient+layer_normalization_7/moments/mean:output:0*
T0*,
_output_shapes
:?????????Θ?
/layer_normalization_7/moments/SquaredDifferenceSquaredDifference tf.__operators__.add_6/AddV2:z:03layer_normalization_7/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θ
8layer_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ξ
&layer_normalization_7/moments/varianceMean3layer_normalization_7/moments/SquaredDifference:z:0Alayer_normalization_7/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(j
%layer_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Δ
#layer_normalization_7/batchnorm/addAddV2/layer_normalization_7/moments/variance:output:0.layer_normalization_7/batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θ
%layer_normalization_7/batchnorm/RsqrtRsqrt'layer_normalization_7/batchnorm/add:z:0*
T0*,
_output_shapes
:?????????Θͺ
2layer_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Θ
#layer_normalization_7/batchnorm/mulMul)layer_normalization_7/batchnorm/Rsqrt:y:0:layer_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ?
%layer_normalization_7/batchnorm/mul_1Mul tf.__operators__.add_6/AddV2:z:0'layer_normalization_7/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ΘΉ
%layer_normalization_7/batchnorm/mul_2Mul+layer_normalization_7/moments/mean:output:0'layer_normalization_7/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θ’
.layer_normalization_7/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Δ
#layer_normalization_7/batchnorm/subSub6layer_normalization_7/batchnorm/ReadVariableOp:value:0)layer_normalization_7/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????ΘΉ
%layer_normalization_7/batchnorm/add_1AddV2)layer_normalization_7/batchnorm/mul_1:z:0'layer_normalization_7/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Θi
conv1d_6/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????·
conv1d_6/Conv1D/ExpandDims
ExpandDims)layer_normalization_7/batchnorm/add_1:z:0'conv1d_6/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ€
+conv1d_6/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_6/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_6/Conv1D/ExpandDims_1
ExpandDims3conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_6/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Ι
conv1d_6/Conv1DConv2D#conv1d_6/Conv1D/ExpandDims:output:0%conv1d_6/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

conv1d_6/Conv1D/SqueezeSqueezeconv1d_6/Conv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????
conv1d_6/BiasAdd/ReadVariableOpReadVariableOp(conv1d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv1d_6/BiasAddBiasAdd conv1d_6/Conv1D/Squeeze:output:0'conv1d_6/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θg
conv1d_6/ReluReluconv1d_6/BiasAdd:output:0*
T0*,
_output_shapes
:?????????Θ\
dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ?
dropout_7/dropout/MulMulconv1d_6/Relu:activations:0 dropout_7/dropout/Const:output:0*
T0*,
_output_shapes
:?????????Θb
dropout_7/dropout/ShapeShapeconv1d_6/Relu:activations:0*
T0*
_output_shapes
:₯
.dropout_7/dropout/random_uniform/RandomUniformRandomUniform dropout_7/dropout/Shape:output:0*
T0*,
_output_shapes
:?????????Θ*
dtype0e
 dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >Ι
dropout_7/dropout/GreaterEqualGreaterEqual7dropout_7/dropout/random_uniform/RandomUniform:output:0)dropout_7/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????Θ
dropout_7/dropout/CastCast"dropout_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????Θ
dropout_7/dropout/Mul_1Muldropout_7/dropout/Mul:z:0dropout_7/dropout/Cast:y:0*
T0*,
_output_shapes
:?????????Θi
conv1d_7/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????©
conv1d_7/Conv1D/ExpandDims
ExpandDimsdropout_7/dropout/Mul_1:z:0'conv1d_7/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ€
+conv1d_7/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_7_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_7/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_7/Conv1D/ExpandDims_1
ExpandDims3conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_7/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Ι
conv1d_7/Conv1DConv2D#conv1d_7/Conv1D/ExpandDims:output:0%conv1d_7/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

conv1d_7/Conv1D/SqueezeSqueezeconv1d_7/Conv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????
conv1d_7/BiasAdd/ReadVariableOpReadVariableOp(conv1d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv1d_7/BiasAddBiasAdd conv1d_7/Conv1D/Squeeze:output:0'conv1d_7/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θ
tf.__operators__.add_7/AddV2AddV2conv1d_7/BiasAdd:output:0 tf.__operators__.add_6/AddV2:z:0*
T0*,
_output_shapes
:?????????Θq
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :΄
global_average_pooling1d/MeanMean tf.__operators__.add_7/AddV2:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:?????????Θ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
Θ*
dtype0
dense/MatMulMatMul&global_average_pooling1d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????\
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUΥ?
dropout_8/dropout/MulMuldense/Relu:activations:0 dropout_8/dropout/Const:output:0*
T0*(
_output_shapes
:?????????_
dropout_8/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:‘
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0e
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ>Ε
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*(
_output_shapes
:?????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_1/MatMulMatMuldropout_8/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_3/BiasAdd/ReadVariableOp,^conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_4/BiasAdd/ReadVariableOp,^conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_5/BiasAdd/ReadVariableOp,^conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_6/BiasAdd/ReadVariableOp,^conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_7/BiasAdd/ReadVariableOp,^conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp/^layer_normalization_4/batchnorm/ReadVariableOp3^layer_normalization_4/batchnorm/mul/ReadVariableOp/^layer_normalization_5/batchnorm/ReadVariableOp3^layer_normalization_5/batchnorm/mul/ReadVariableOp/^layer_normalization_6/batchnorm/ReadVariableOp3^layer_normalization_6/batchnorm/mul/ReadVariableOp/^layer_normalization_7/batchnorm/ReadVariableOp3^layer_normalization_7/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp;^multi_head_attention_1/attention_output/add/ReadVariableOpE^multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_1/key/add/ReadVariableOp8^multi_head_attention_1/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/query/add/ReadVariableOp:^multi_head_attention_1/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/value/add/ReadVariableOp:^multi_head_attention_1/value/einsum/Einsum/ReadVariableOp;^multi_head_attention_2/attention_output/add/ReadVariableOpE^multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_2/key/add/ReadVariableOp8^multi_head_attention_2/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_2/query/add/ReadVariableOp:^multi_head_attention_2/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_2/value/add/ReadVariableOp:^multi_head_attention_2/value/einsum/Einsum/ReadVariableOp;^multi_head_attention_3/attention_output/add/ReadVariableOpE^multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_3/key/add/ReadVariableOp8^multi_head_attention_3/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_3/query/add/ReadVariableOp:^multi_head_attention_3/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_3/value/add/ReadVariableOp:^multi_head_attention_3/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*΅
_input_shapes£
 :?????????Θ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2Z
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_3/BiasAdd/ReadVariableOpconv1d_3/BiasAdd/ReadVariableOp2Z
+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_4/BiasAdd/ReadVariableOpconv1d_4/BiasAdd/ReadVariableOp2Z
+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_5/BiasAdd/ReadVariableOpconv1d_5/BiasAdd/ReadVariableOp2Z
+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_6/BiasAdd/ReadVariableOpconv1d_6/BiasAdd/ReadVariableOp2Z
+conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_7/BiasAdd/ReadVariableOpconv1d_7/BiasAdd/ReadVariableOp2Z
+conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2`
.layer_normalization_2/batchnorm/ReadVariableOp.layer_normalization_2/batchnorm/ReadVariableOp2h
2layer_normalization_2/batchnorm/mul/ReadVariableOp2layer_normalization_2/batchnorm/mul/ReadVariableOp2`
.layer_normalization_3/batchnorm/ReadVariableOp.layer_normalization_3/batchnorm/ReadVariableOp2h
2layer_normalization_3/batchnorm/mul/ReadVariableOp2layer_normalization_3/batchnorm/mul/ReadVariableOp2`
.layer_normalization_4/batchnorm/ReadVariableOp.layer_normalization_4/batchnorm/ReadVariableOp2h
2layer_normalization_4/batchnorm/mul/ReadVariableOp2layer_normalization_4/batchnorm/mul/ReadVariableOp2`
.layer_normalization_5/batchnorm/ReadVariableOp.layer_normalization_5/batchnorm/ReadVariableOp2h
2layer_normalization_5/batchnorm/mul/ReadVariableOp2layer_normalization_5/batchnorm/mul/ReadVariableOp2`
.layer_normalization_6/batchnorm/ReadVariableOp.layer_normalization_6/batchnorm/ReadVariableOp2h
2layer_normalization_6/batchnorm/mul/ReadVariableOp2layer_normalization_6/batchnorm/mul/ReadVariableOp2`
.layer_normalization_7/batchnorm/ReadVariableOp.layer_normalization_7/batchnorm/ReadVariableOp2h
2layer_normalization_7/batchnorm/mul/ReadVariableOp2layer_normalization_7/batchnorm/mul/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp2x
:multi_head_attention_1/attention_output/add/ReadVariableOp:multi_head_attention_1/attention_output/add/ReadVariableOp2
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_1/key/add/ReadVariableOp-multi_head_attention_1/key/add/ReadVariableOp2r
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_1/query/add/ReadVariableOp/multi_head_attention_1/query/add/ReadVariableOp2v
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_1/value/add/ReadVariableOp/multi_head_attention_1/value/add/ReadVariableOp2v
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp2x
:multi_head_attention_2/attention_output/add/ReadVariableOp:multi_head_attention_2/attention_output/add/ReadVariableOp2
Dmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_2/key/add/ReadVariableOp-multi_head_attention_2/key/add/ReadVariableOp2r
7multi_head_attention_2/key/einsum/Einsum/ReadVariableOp7multi_head_attention_2/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_2/query/add/ReadVariableOp/multi_head_attention_2/query/add/ReadVariableOp2v
9multi_head_attention_2/query/einsum/Einsum/ReadVariableOp9multi_head_attention_2/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_2/value/add/ReadVariableOp/multi_head_attention_2/value/add/ReadVariableOp2v
9multi_head_attention_2/value/einsum/Einsum/ReadVariableOp9multi_head_attention_2/value/einsum/Einsum/ReadVariableOp2x
:multi_head_attention_3/attention_output/add/ReadVariableOp:multi_head_attention_3/attention_output/add/ReadVariableOp2
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_3/key/add/ReadVariableOp-multi_head_attention_3/key/add/ReadVariableOp2r
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_3/query/add/ReadVariableOp/multi_head_attention_3/query/add/ReadVariableOp2v
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_3/value/add/ReadVariableOp/multi_head_attention_3/value/add/ReadVariableOp2v
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
λ
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_13802

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:?????????Θ`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:?????????Θ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
·2
ϊ
Q__inference_multi_head_attention_3_layer_call_and_return_conditional_losses_18357	
query	
valueA
+query_einsum_einsum_readvariableop_resource:23
!query_add_readvariableop_resource:2?
)key_einsum_einsum_readvariableop_resource:21
key_add_readvariableop_resource:2A
+value_einsum_einsum_readvariableop_resource:23
!value_add_readvariableop_resource:2L
6attention_output_einsum_einsum_readvariableop_resource:2:
,attention_output_add_readvariableop_resource:
identity’#attention_output/add/ReadVariableOp’-attention_output/einsum/Einsum/ReadVariableOp’key/add/ReadVariableOp’ key/einsum/Einsum/ReadVariableOp’query/add/ReadVariableOp’"query/einsum/Einsum/ReadVariableOp’value/add/ReadVariableOp’"value/einsum/Einsum/ReadVariableOp
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:2*
dtype0
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0­
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:2*
dtype0
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:2*
dtype0
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΓΠ>d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:?????????Θ2
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:?????????ΘΘ*
equationaecd,abcd->acbep
softmax_3/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:?????????ΘΘ]
dropout_12/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ?
dropout_12/dropout/MulMulsoftmax_3/Softmax:softmax:0!dropout_12/dropout/Const:output:0*
T0*1
_output_shapes
:?????????ΘΘc
dropout_12/dropout/ShapeShapesoftmax_3/Softmax:softmax:0*
T0*
_output_shapes
:¬
/dropout_12/dropout/random_uniform/RandomUniformRandomUniform!dropout_12/dropout/Shape:output:0*
T0*1
_output_shapes
:?????????ΘΘ*
dtype0f
!dropout_12/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >Ρ
dropout_12/dropout/GreaterEqualGreaterEqual8dropout_12/dropout/random_uniform/RandomUniform:output:0*dropout_12/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:?????????ΘΘ
dropout_12/dropout/CastCast#dropout_12/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:?????????ΘΘ
dropout_12/dropout/Mul_1Muldropout_12/dropout/Mul:z:0dropout_12/dropout/Cast:y:0*
T0*1
_output_shapes
:?????????ΘΘ©
einsum_1/EinsumEinsumdropout_12/dropout/Mul_1:z:0value/add:z:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationacbe,aecd->abcd¨
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0Φ
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????Θ*
equationabcd,cde->abe
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0ͺ
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θl
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:?????????ΘΨ
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Θ:?????????Θ: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:?????????Θ

_user_specified_namequery:SO
,
_output_shapes
:?????????Θ

_user_specified_namevalue
λ
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_13970

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:?????????Θ`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:?????????Θ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
φ

C__inference_conv1d_1_layer_call_and_return_conditional_losses_13651

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θd
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
ϋ

N__inference_layer_normalization_layer_call_and_return_conditional_losses_13512

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(r
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:?????????Θ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¬
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θb
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:?????????Θ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θh
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θw
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????Θw
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Θg
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:?????????Θ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs


c
D__inference_dropout_5_layer_call_and_return_conditional_losses_14589

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:?????????ΘC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:?????????Θ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????Θt
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????Θn
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:?????????Θ^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:?????????Θ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs

b
)__inference_dropout_1_layer_call_fn_17592

inputs
identity’StatefulPartitionedCallΔ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_14935t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
½*
ϊ
Q__inference_multi_head_attention_3_layer_call_and_return_conditional_losses_18315	
query	
valueA
+query_einsum_einsum_readvariableop_resource:23
!query_add_readvariableop_resource:2?
)key_einsum_einsum_readvariableop_resource:21
key_add_readvariableop_resource:2A
+value_einsum_einsum_readvariableop_resource:23
!value_add_readvariableop_resource:2L
6attention_output_einsum_einsum_readvariableop_resource:2:
,attention_output_add_readvariableop_resource:
identity’#attention_output/add/ReadVariableOp’-attention_output/einsum/Einsum/ReadVariableOp’key/add/ReadVariableOp’ key/einsum/Einsum/ReadVariableOp’query/add/ReadVariableOp’"query/einsum/Einsum/ReadVariableOp’value/add/ReadVariableOp’"value/einsum/Einsum/ReadVariableOp
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:2*
dtype0
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0­
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:2*
dtype0
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:2*
dtype0
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΓΠ>d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:?????????Θ2
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:?????????ΘΘ*
equationaecd,abcd->acbep
softmax_3/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:?????????ΘΘx
dropout_12/IdentityIdentitysoftmax_3/Softmax:softmax:0*
T0*1
_output_shapes
:?????????ΘΘ©
einsum_1/EinsumEinsumdropout_12/Identity:output:0value/add:z:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationacbe,aecd->abcd¨
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0Φ
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????Θ*
equationabcd,cde->abe
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0ͺ
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θl
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:?????????ΘΨ
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Θ:?????????Θ: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:?????????Θ

_user_specified_namequery:SO
,
_output_shapes
:?????????Θ

_user_specified_namevalue

ι
4__inference_multi_head_attention_layer_call_fn_17400	
query	
value
unknown:2
	unknown_0:2
	unknown_1:2
	unknown_2:2
	unknown_3:2
	unknown_4:2
	unknown_5:2
	unknown_6:
identity’StatefulPartitionedCallΎ
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_13553t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Θ:?????????Θ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:?????????Θ

_user_specified_namequery:SO
,
_output_shapes
:?????????Θ

_user_specified_namevalue
½*
ϊ
Q__inference_multi_head_attention_3_layer_call_and_return_conditional_losses_14057	
query	
valueA
+query_einsum_einsum_readvariableop_resource:23
!query_add_readvariableop_resource:2?
)key_einsum_einsum_readvariableop_resource:21
key_add_readvariableop_resource:2A
+value_einsum_einsum_readvariableop_resource:23
!value_add_readvariableop_resource:2L
6attention_output_einsum_einsum_readvariableop_resource:2:
,attention_output_add_readvariableop_resource:
identity’#attention_output/add/ReadVariableOp’-attention_output/einsum/Einsum/ReadVariableOp’key/add/ReadVariableOp’ key/einsum/Einsum/ReadVariableOp’query/add/ReadVariableOp’"query/einsum/Einsum/ReadVariableOp’value/add/ReadVariableOp’"value/einsum/Einsum/ReadVariableOp
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:2*
dtype0
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0­
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:2*
dtype0
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:2*
dtype0
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΓΠ>d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:?????????Θ2
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:?????????ΘΘ*
equationaecd,abcd->acbep
softmax_3/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:?????????ΘΘx
dropout_12/IdentityIdentitysoftmax_3/Softmax:softmax:0*
T0*1
_output_shapes
:?????????ΘΘ©
einsum_1/EinsumEinsumdropout_12/Identity:output:0value/add:z:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationacbe,aecd->abcd¨
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0Φ
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????Θ*
equationabcd,cde->abe
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0ͺ
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θl
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:?????????ΘΨ
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Θ:?????????Θ: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:?????????Θ

_user_specified_namequery:SO
,
_output_shapes
:?????????Θ

_user_specified_namevalue
±
E
)__inference_dropout_5_layer_call_fn_18159

inputs
identity΄
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_13970e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:?????????Θ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
Ν

A__inference_conv1d_layer_call_and_return_conditional_losses_17582

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????ΘU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:?????????Θf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:?????????Θ
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
Ψ

(__inference_conv1d_6_layer_call_fn_18424

inputs
unknown:
	unknown_0:
identity’StatefulPartitionedCallέ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_6_layer_call_and_return_conditional_losses_14127t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
Ψ

(__inference_conv1d_3_layer_call_fn_17904

inputs
unknown:
	unknown_0:
identity’StatefulPartitionedCallέ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_3_layer_call_and_return_conditional_losses_13819t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
±
E
)__inference_dropout_4_layer_call_fn_18076

inputs
identity΄
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_13912e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:?????????Θ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs

λ
6__inference_multi_head_attention_1_layer_call_fn_17686	
query	
value
unknown:2
	unknown_0:2
	unknown_1:2
	unknown_2:2
	unknown_3:2
	unknown_4:2
	unknown_5:2
	unknown_6:
identity’StatefulPartitionedCallΐ
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_13721t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Θ:?????????Θ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:?????????Θ

_user_specified_namequery:SO
,
_output_shapes
:?????????Θ

_user_specified_namevalue
κ

5__inference_layer_normalization_5_layer_call_fn_18107

inputs
unknown:
	unknown_0:
identity’StatefulPartitionedCallκ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_13937t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
β
΄
#__inference_signature_wrapper_16190
input_1
unknown:
	unknown_0:
	unknown_1:2
	unknown_2:2
	unknown_3:2
	unknown_4:2
	unknown_5:2
	unknown_6:2
	unknown_7:2
	unknown_8:
	unknown_9:

unknown_10: 

unknown_11:

unknown_12: 

unknown_13:

unknown_14:

unknown_15:

unknown_16: 

unknown_17:2

unknown_18:2 

unknown_19:2

unknown_20:2 

unknown_21:2

unknown_22:2 

unknown_23:2

unknown_24:

unknown_25:

unknown_26: 

unknown_27:

unknown_28: 

unknown_29:

unknown_30:

unknown_31:

unknown_32: 

unknown_33:2

unknown_34:2 

unknown_35:2

unknown_36:2 

unknown_37:2

unknown_38:2 

unknown_39:2

unknown_40:

unknown_41:

unknown_42: 

unknown_43:

unknown_44: 

unknown_45:

unknown_46:

unknown_47:

unknown_48: 

unknown_49:2

unknown_50:2 

unknown_51:2

unknown_52:2 

unknown_53:2

unknown_54:2 

unknown_55:2

unknown_56:

unknown_57:

unknown_58: 

unknown_59:

unknown_60: 

unknown_61:

unknown_62:

unknown_63:
Θ

unknown_64:	

unknown_65:	

unknown_66:
identity’StatefulPartitionedCallΙ	
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66*P
TinI
G2E*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*f
_read_only_resource_inputsH
FD	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCD*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_13470o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*΅
_input_shapes£
 :?????????Θ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:?????????Θ
!
_user_specified_name	input_1
ι
`
B__inference_dropout_layer_call_and_return_conditional_losses_17514

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:?????????Θ`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:?????????Θ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
Ν

A__inference_conv1d_layer_call_and_return_conditional_losses_13623

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????ΘU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:?????????Θf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:?????????Θ
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
λ
b
D__inference_dropout_6_layer_call_and_return_conditional_losses_18372

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:?????????Θ`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:?????????Θ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
Ο

C__inference_conv1d_6_layer_call_and_return_conditional_losses_18440

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????ΘU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:?????????Θf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:?????????Θ
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
Ϋ
b
D__inference_dropout_8_layer_call_and_return_conditional_losses_14185

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:?????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs

΅
%__inference_model_layer_call_fn_16331

inputs
unknown:
	unknown_0:
	unknown_1:2
	unknown_2:2
	unknown_3:2
	unknown_4:2
	unknown_5:2
	unknown_6:2
	unknown_7:2
	unknown_8:
	unknown_9:

unknown_10: 

unknown_11:

unknown_12: 

unknown_13:

unknown_14:

unknown_15:

unknown_16: 

unknown_17:2

unknown_18:2 

unknown_19:2

unknown_20:2 

unknown_21:2

unknown_22:2 

unknown_23:2

unknown_24:

unknown_25:

unknown_26: 

unknown_27:

unknown_28: 

unknown_29:

unknown_30:

unknown_31:

unknown_32: 

unknown_33:2

unknown_34:2 

unknown_35:2

unknown_36:2 

unknown_37:2

unknown_38:2 

unknown_39:2

unknown_40:

unknown_41:

unknown_42: 

unknown_43:

unknown_44: 

unknown_45:

unknown_46:

unknown_47:

unknown_48: 

unknown_49:2

unknown_50:2 

unknown_51:2

unknown_52:2 

unknown_53:2

unknown_54:2 

unknown_55:2

unknown_56:

unknown_57:

unknown_58: 

unknown_59:

unknown_60: 

unknown_61:

unknown_62:

unknown_63:
Θ

unknown_64:	

unknown_65:	

unknown_66:
identity’StatefulPartitionedCallθ	
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66*P
TinI
G2E*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*f
_read_only_resource_inputsH
FD	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCD*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_14204o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*΅
_input_shapes£
 :?????????Θ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs

o
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_13480

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs

b
)__inference_dropout_6_layer_call_fn_18367

inputs
identity’StatefulPartitionedCallΔ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_14459t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
Α

%__inference_dense_layer_call_fn_18511

inputs
unknown:
Θ
	unknown_0:	
identity’StatefulPartitionedCallΦ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_14174p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
·2
ϊ
Q__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_14703	
query	
valueA
+query_einsum_einsum_readvariableop_resource:23
!query_add_readvariableop_resource:2?
)key_einsum_einsum_readvariableop_resource:21
key_add_readvariableop_resource:2A
+value_einsum_einsum_readvariableop_resource:23
!value_add_readvariableop_resource:2L
6attention_output_einsum_einsum_readvariableop_resource:2:
,attention_output_add_readvariableop_resource:
identity’#attention_output/add/ReadVariableOp’-attention_output/einsum/Einsum/ReadVariableOp’key/add/ReadVariableOp’ key/einsum/Einsum/ReadVariableOp’query/add/ReadVariableOp’"query/einsum/Einsum/ReadVariableOp’value/add/ReadVariableOp’"value/einsum/Einsum/ReadVariableOp
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:2*
dtype0
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0­
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:2*
dtype0
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:2*
dtype0
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΓΠ>d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:?????????Θ2
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:?????????ΘΘ*
equationaecd,abcd->acbep
softmax_2/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:?????????ΘΘ]
dropout_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ?
dropout_11/dropout/MulMulsoftmax_2/Softmax:softmax:0!dropout_11/dropout/Const:output:0*
T0*1
_output_shapes
:?????????ΘΘc
dropout_11/dropout/ShapeShapesoftmax_2/Softmax:softmax:0*
T0*
_output_shapes
:¬
/dropout_11/dropout/random_uniform/RandomUniformRandomUniform!dropout_11/dropout/Shape:output:0*
T0*1
_output_shapes
:?????????ΘΘ*
dtype0f
!dropout_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >Ρ
dropout_11/dropout/GreaterEqualGreaterEqual8dropout_11/dropout/random_uniform/RandomUniform:output:0*dropout_11/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:?????????ΘΘ
dropout_11/dropout/CastCast#dropout_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:?????????ΘΘ
dropout_11/dropout/Mul_1Muldropout_11/dropout/Mul:z:0dropout_11/dropout/Cast:y:0*
T0*1
_output_shapes
:?????????ΘΘ©
einsum_1/EinsumEinsumdropout_11/dropout/Mul_1:z:0value/add:z:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationacbe,aecd->abcd¨
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0Φ
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????Θ*
equationabcd,cde->abe
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0ͺ
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θl
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:?????????ΘΨ
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Θ:?????????Θ: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:?????????Θ

_user_specified_namequery:SO
,
_output_shapes
:?????????Θ

_user_specified_namevalue
ϊ	
c
D__inference_dropout_8_layer_call_and_return_conditional_losses_18549

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUΥ?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ>§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:?????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs


c
D__inference_dropout_3_layer_call_and_return_conditional_losses_14762

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:?????????ΘC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:?????????Θ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????Θt
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????Θn
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:?????????Θ^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:?????????Θ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
½*
ϊ
Q__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_18029	
query	
valueA
+query_einsum_einsum_readvariableop_resource:23
!query_add_readvariableop_resource:2?
)key_einsum_einsum_readvariableop_resource:21
key_add_readvariableop_resource:2A
+value_einsum_einsum_readvariableop_resource:23
!value_add_readvariableop_resource:2L
6attention_output_einsum_einsum_readvariableop_resource:2:
,attention_output_add_readvariableop_resource:
identity’#attention_output/add/ReadVariableOp’-attention_output/einsum/Einsum/ReadVariableOp’key/add/ReadVariableOp’ key/einsum/Einsum/ReadVariableOp’query/add/ReadVariableOp’"query/einsum/Einsum/ReadVariableOp’value/add/ReadVariableOp’"value/einsum/Einsum/ReadVariableOp
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:2*
dtype0
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0­
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:2*
dtype0
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:2*
dtype0
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΓΠ>d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:?????????Θ2
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:?????????ΘΘ*
equationaecd,abcd->acbep
softmax_2/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:?????????ΘΘx
dropout_11/IdentityIdentitysoftmax_2/Softmax:softmax:0*
T0*1
_output_shapes
:?????????ΘΘ©
einsum_1/EinsumEinsumdropout_11/Identity:output:0value/add:z:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationacbe,aecd->abcd¨
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0Φ
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????Θ*
equationabcd,cde->abe
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0ͺ
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θl
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:?????????ΘΨ
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Θ:?????????Θ: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:?????????Θ

_user_specified_namequery:SO
,
_output_shapes
:?????????Θ

_user_specified_namevalue
ύ

P__inference_layer_normalization_6_layer_call_and_return_conditional_losses_14016

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(r
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:?????????Θ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¬
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θb
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:?????????Θ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θh
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θw
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????Θw
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Θg
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:?????????Θ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
λ
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_18169

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:?????????Θ`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:?????????Θ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
ι
`
B__inference_dropout_layer_call_and_return_conditional_losses_13576

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:?????????Θ`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:?????????Θ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
2
ψ
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_17499	
query	
valueA
+query_einsum_einsum_readvariableop_resource:23
!query_add_readvariableop_resource:2?
)key_einsum_einsum_readvariableop_resource:21
key_add_readvariableop_resource:2A
+value_einsum_einsum_readvariableop_resource:23
!value_add_readvariableop_resource:2L
6attention_output_einsum_einsum_readvariableop_resource:2:
,attention_output_add_readvariableop_resource:
identity’#attention_output/add/ReadVariableOp’-attention_output/einsum/Einsum/ReadVariableOp’key/add/ReadVariableOp’ key/einsum/Einsum/ReadVariableOp’query/add/ReadVariableOp’"query/einsum/Einsum/ReadVariableOp’value/add/ReadVariableOp’"value/einsum/Einsum/ReadVariableOp
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:2*
dtype0
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0­
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:2*
dtype0
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:2*
dtype0
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΓΠ>d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:?????????Θ2
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:?????????ΘΘ*
equationaecd,abcd->acben
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:?????????ΘΘ\
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ?
dropout_9/dropout/MulMulsoftmax/Softmax:softmax:0 dropout_9/dropout/Const:output:0*
T0*1
_output_shapes
:?????????ΘΘ`
dropout_9/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:ͺ
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*1
_output_shapes
:?????????ΘΘ*
dtype0e
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >Ξ
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:?????????ΘΘ
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:?????????ΘΘ
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*1
_output_shapes
:?????????ΘΘ¨
einsum_1/EinsumEinsumdropout_9/dropout/Mul_1:z:0value/add:z:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationacbe,aecd->abcd¨
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0Φ
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????Θ*
equationabcd,cde->abe
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0ͺ
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θl
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:?????????ΘΨ
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Θ:?????????Θ: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:?????????Θ

_user_specified_namequery:SO
,
_output_shapes
:?????????Θ

_user_specified_namevalue
ξΚ
?e
!__inference__traced_restore_19471
file_prefix8
*assignvariableop_layer_normalization_gamma:9
+assignvariableop_1_layer_normalization_beta:<
.assignvariableop_2_layer_normalization_1_gamma:;
-assignvariableop_3_layer_normalization_1_beta:6
 assignvariableop_4_conv1d_kernel:,
assignvariableop_5_conv1d_bias:8
"assignvariableop_6_conv1d_1_kernel:.
 assignvariableop_7_conv1d_1_bias:<
.assignvariableop_8_layer_normalization_2_gamma:;
-assignvariableop_9_layer_normalization_2_beta:=
/assignvariableop_10_layer_normalization_3_gamma:<
.assignvariableop_11_layer_normalization_3_beta:9
#assignvariableop_12_conv1d_2_kernel:/
!assignvariableop_13_conv1d_2_bias:9
#assignvariableop_14_conv1d_3_kernel:/
!assignvariableop_15_conv1d_3_bias:=
/assignvariableop_16_layer_normalization_4_gamma:<
.assignvariableop_17_layer_normalization_4_beta:=
/assignvariableop_18_layer_normalization_5_gamma:<
.assignvariableop_19_layer_normalization_5_beta:9
#assignvariableop_20_conv1d_4_kernel:/
!assignvariableop_21_conv1d_4_bias:9
#assignvariableop_22_conv1d_5_kernel:/
!assignvariableop_23_conv1d_5_bias:=
/assignvariableop_24_layer_normalization_6_gamma:<
.assignvariableop_25_layer_normalization_6_beta:=
/assignvariableop_26_layer_normalization_7_gamma:<
.assignvariableop_27_layer_normalization_7_beta:9
#assignvariableop_28_conv1d_6_kernel:/
!assignvariableop_29_conv1d_6_bias:9
#assignvariableop_30_conv1d_7_kernel:/
!assignvariableop_31_conv1d_7_bias:4
 assignvariableop_32_dense_kernel:
Θ-
assignvariableop_33_dense_bias:	5
"assignvariableop_34_dense_1_kernel:	.
 assignvariableop_35_dense_1_bias:K
5assignvariableop_36_multi_head_attention_query_kernel:2E
3assignvariableop_37_multi_head_attention_query_bias:2I
3assignvariableop_38_multi_head_attention_key_kernel:2C
1assignvariableop_39_multi_head_attention_key_bias:2K
5assignvariableop_40_multi_head_attention_value_kernel:2E
3assignvariableop_41_multi_head_attention_value_bias:2V
@assignvariableop_42_multi_head_attention_attention_output_kernel:2L
>assignvariableop_43_multi_head_attention_attention_output_bias:M
7assignvariableop_44_multi_head_attention_1_query_kernel:2G
5assignvariableop_45_multi_head_attention_1_query_bias:2K
5assignvariableop_46_multi_head_attention_1_key_kernel:2E
3assignvariableop_47_multi_head_attention_1_key_bias:2M
7assignvariableop_48_multi_head_attention_1_value_kernel:2G
5assignvariableop_49_multi_head_attention_1_value_bias:2X
Bassignvariableop_50_multi_head_attention_1_attention_output_kernel:2N
@assignvariableop_51_multi_head_attention_1_attention_output_bias:M
7assignvariableop_52_multi_head_attention_2_query_kernel:2G
5assignvariableop_53_multi_head_attention_2_query_bias:2K
5assignvariableop_54_multi_head_attention_2_key_kernel:2E
3assignvariableop_55_multi_head_attention_2_key_bias:2M
7assignvariableop_56_multi_head_attention_2_value_kernel:2G
5assignvariableop_57_multi_head_attention_2_value_bias:2X
Bassignvariableop_58_multi_head_attention_2_attention_output_kernel:2N
@assignvariableop_59_multi_head_attention_2_attention_output_bias:M
7assignvariableop_60_multi_head_attention_3_query_kernel:2G
5assignvariableop_61_multi_head_attention_3_query_bias:2K
5assignvariableop_62_multi_head_attention_3_key_kernel:2E
3assignvariableop_63_multi_head_attention_3_key_bias:2M
7assignvariableop_64_multi_head_attention_3_value_kernel:2G
5assignvariableop_65_multi_head_attention_3_value_bias:2X
Bassignvariableop_66_multi_head_attention_3_attention_output_kernel:2N
@assignvariableop_67_multi_head_attention_3_attention_output_bias:"
assignvariableop_68_iter:	 #
assignvariableop_69_decay: +
!assignvariableop_70_learning_rate: &
assignvariableop_71_momentum: !
assignvariableop_72_rho: %
assignvariableop_73_total_1: %
assignvariableop_74_count_1: #
assignvariableop_75_total: #
assignvariableop_76_count: ?
1assignvariableop_77_layer_normalization_gamma_rms:>
0assignvariableop_78_layer_normalization_beta_rms:A
3assignvariableop_79_layer_normalization_1_gamma_rms:@
2assignvariableop_80_layer_normalization_1_beta_rms:;
%assignvariableop_81_conv1d_kernel_rms:1
#assignvariableop_82_conv1d_bias_rms:=
'assignvariableop_83_conv1d_1_kernel_rms:3
%assignvariableop_84_conv1d_1_bias_rms:A
3assignvariableop_85_layer_normalization_2_gamma_rms:@
2assignvariableop_86_layer_normalization_2_beta_rms:A
3assignvariableop_87_layer_normalization_3_gamma_rms:@
2assignvariableop_88_layer_normalization_3_beta_rms:=
'assignvariableop_89_conv1d_2_kernel_rms:3
%assignvariableop_90_conv1d_2_bias_rms:=
'assignvariableop_91_conv1d_3_kernel_rms:3
%assignvariableop_92_conv1d_3_bias_rms:A
3assignvariableop_93_layer_normalization_4_gamma_rms:@
2assignvariableop_94_layer_normalization_4_beta_rms:A
3assignvariableop_95_layer_normalization_5_gamma_rms:@
2assignvariableop_96_layer_normalization_5_beta_rms:=
'assignvariableop_97_conv1d_4_kernel_rms:3
%assignvariableop_98_conv1d_4_bias_rms:=
'assignvariableop_99_conv1d_5_kernel_rms:4
&assignvariableop_100_conv1d_5_bias_rms:B
4assignvariableop_101_layer_normalization_6_gamma_rms:A
3assignvariableop_102_layer_normalization_6_beta_rms:B
4assignvariableop_103_layer_normalization_7_gamma_rms:A
3assignvariableop_104_layer_normalization_7_beta_rms:>
(assignvariableop_105_conv1d_6_kernel_rms:4
&assignvariableop_106_conv1d_6_bias_rms:>
(assignvariableop_107_conv1d_7_kernel_rms:4
&assignvariableop_108_conv1d_7_bias_rms:9
%assignvariableop_109_dense_kernel_rms:
Θ2
#assignvariableop_110_dense_bias_rms:	:
'assignvariableop_111_dense_1_kernel_rms:	3
%assignvariableop_112_dense_1_bias_rms:P
:assignvariableop_113_multi_head_attention_query_kernel_rms:2J
8assignvariableop_114_multi_head_attention_query_bias_rms:2N
8assignvariableop_115_multi_head_attention_key_kernel_rms:2H
6assignvariableop_116_multi_head_attention_key_bias_rms:2P
:assignvariableop_117_multi_head_attention_value_kernel_rms:2J
8assignvariableop_118_multi_head_attention_value_bias_rms:2[
Eassignvariableop_119_multi_head_attention_attention_output_kernel_rms:2Q
Cassignvariableop_120_multi_head_attention_attention_output_bias_rms:R
<assignvariableop_121_multi_head_attention_1_query_kernel_rms:2L
:assignvariableop_122_multi_head_attention_1_query_bias_rms:2P
:assignvariableop_123_multi_head_attention_1_key_kernel_rms:2J
8assignvariableop_124_multi_head_attention_1_key_bias_rms:2R
<assignvariableop_125_multi_head_attention_1_value_kernel_rms:2L
:assignvariableop_126_multi_head_attention_1_value_bias_rms:2]
Gassignvariableop_127_multi_head_attention_1_attention_output_kernel_rms:2S
Eassignvariableop_128_multi_head_attention_1_attention_output_bias_rms:R
<assignvariableop_129_multi_head_attention_2_query_kernel_rms:2L
:assignvariableop_130_multi_head_attention_2_query_bias_rms:2P
:assignvariableop_131_multi_head_attention_2_key_kernel_rms:2J
8assignvariableop_132_multi_head_attention_2_key_bias_rms:2R
<assignvariableop_133_multi_head_attention_2_value_kernel_rms:2L
:assignvariableop_134_multi_head_attention_2_value_bias_rms:2]
Gassignvariableop_135_multi_head_attention_2_attention_output_kernel_rms:2S
Eassignvariableop_136_multi_head_attention_2_attention_output_bias_rms:R
<assignvariableop_137_multi_head_attention_3_query_kernel_rms:2L
:assignvariableop_138_multi_head_attention_3_query_bias_rms:2P
:assignvariableop_139_multi_head_attention_3_key_kernel_rms:2J
8assignvariableop_140_multi_head_attention_3_key_bias_rms:2R
<assignvariableop_141_multi_head_attention_3_value_kernel_rms:2L
:assignvariableop_142_multi_head_attention_3_value_bias_rms:2]
Gassignvariableop_143_multi_head_attention_3_attention_output_kernel_rms:2S
Eassignvariableop_144_multi_head_attention_3_attention_output_bias_rms:
identity_146’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_100’AssignVariableOp_101’AssignVariableOp_102’AssignVariableOp_103’AssignVariableOp_104’AssignVariableOp_105’AssignVariableOp_106’AssignVariableOp_107’AssignVariableOp_108’AssignVariableOp_109’AssignVariableOp_11’AssignVariableOp_110’AssignVariableOp_111’AssignVariableOp_112’AssignVariableOp_113’AssignVariableOp_114’AssignVariableOp_115’AssignVariableOp_116’AssignVariableOp_117’AssignVariableOp_118’AssignVariableOp_119’AssignVariableOp_12’AssignVariableOp_120’AssignVariableOp_121’AssignVariableOp_122’AssignVariableOp_123’AssignVariableOp_124’AssignVariableOp_125’AssignVariableOp_126’AssignVariableOp_127’AssignVariableOp_128’AssignVariableOp_129’AssignVariableOp_13’AssignVariableOp_130’AssignVariableOp_131’AssignVariableOp_132’AssignVariableOp_133’AssignVariableOp_134’AssignVariableOp_135’AssignVariableOp_136’AssignVariableOp_137’AssignVariableOp_138’AssignVariableOp_139’AssignVariableOp_14’AssignVariableOp_140’AssignVariableOp_141’AssignVariableOp_142’AssignVariableOp_143’AssignVariableOp_144’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_22’AssignVariableOp_23’AssignVariableOp_24’AssignVariableOp_25’AssignVariableOp_26’AssignVariableOp_27’AssignVariableOp_28’AssignVariableOp_29’AssignVariableOp_3’AssignVariableOp_30’AssignVariableOp_31’AssignVariableOp_32’AssignVariableOp_33’AssignVariableOp_34’AssignVariableOp_35’AssignVariableOp_36’AssignVariableOp_37’AssignVariableOp_38’AssignVariableOp_39’AssignVariableOp_4’AssignVariableOp_40’AssignVariableOp_41’AssignVariableOp_42’AssignVariableOp_43’AssignVariableOp_44’AssignVariableOp_45’AssignVariableOp_46’AssignVariableOp_47’AssignVariableOp_48’AssignVariableOp_49’AssignVariableOp_5’AssignVariableOp_50’AssignVariableOp_51’AssignVariableOp_52’AssignVariableOp_53’AssignVariableOp_54’AssignVariableOp_55’AssignVariableOp_56’AssignVariableOp_57’AssignVariableOp_58’AssignVariableOp_59’AssignVariableOp_6’AssignVariableOp_60’AssignVariableOp_61’AssignVariableOp_62’AssignVariableOp_63’AssignVariableOp_64’AssignVariableOp_65’AssignVariableOp_66’AssignVariableOp_67’AssignVariableOp_68’AssignVariableOp_69’AssignVariableOp_7’AssignVariableOp_70’AssignVariableOp_71’AssignVariableOp_72’AssignVariableOp_73’AssignVariableOp_74’AssignVariableOp_75’AssignVariableOp_76’AssignVariableOp_77’AssignVariableOp_78’AssignVariableOp_79’AssignVariableOp_8’AssignVariableOp_80’AssignVariableOp_81’AssignVariableOp_82’AssignVariableOp_83’AssignVariableOp_84’AssignVariableOp_85’AssignVariableOp_86’AssignVariableOp_87’AssignVariableOp_88’AssignVariableOp_89’AssignVariableOp_9’AssignVariableOp_90’AssignVariableOp_91’AssignVariableOp_92’AssignVariableOp_93’AssignVariableOp_94’AssignVariableOp_95’AssignVariableOp_96’AssignVariableOp_97’AssignVariableOp_98’AssignVariableOp_99H
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*¨G
valueGBGB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/50/.ATTRIBUTES/VARIABLE_VALUEB'variables/51/.ATTRIBUTES/VARIABLE_VALUEB'variables/52/.ATTRIBUTES/VARIABLE_VALUEB'variables/53/.ATTRIBUTES/VARIABLE_VALUEB'variables/54/.ATTRIBUTES/VARIABLE_VALUEB'variables/55/.ATTRIBUTES/VARIABLE_VALUEB'variables/56/.ATTRIBUTES/VARIABLE_VALUEB'variables/57/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-17/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/8/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/9/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/18/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/19/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/20/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/21/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/22/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/23/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/24/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/25/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/34/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/35/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/36/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/37/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/38/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/39/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/40/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/41/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/50/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/51/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/52/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/53/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/54/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/55/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/56/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/57/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*Ί
value°B­B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ή
_output_shapesΛ
Θ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*£
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp*assignvariableop_layer_normalization_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp+assignvariableop_1_layer_normalization_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp.assignvariableop_2_layer_normalization_1_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp-assignvariableop_3_layer_normalization_1_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp assignvariableop_4_conv1d_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_conv1d_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv1d_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv1d_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp.assignvariableop_8_layer_normalization_2_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp-assignvariableop_9_layer_normalization_2_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_10AssignVariableOp/assignvariableop_10_layer_normalization_3_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp.assignvariableop_11_layer_normalization_3_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv1d_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv1d_2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp#assignvariableop_14_conv1d_3_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp!assignvariableop_15_conv1d_3_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_16AssignVariableOp/assignvariableop_16_layer_normalization_4_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp.assignvariableop_17_layer_normalization_4_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_18AssignVariableOp/assignvariableop_18_layer_normalization_5_gammaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp.assignvariableop_19_layer_normalization_5_betaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp#assignvariableop_20_conv1d_4_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp!assignvariableop_21_conv1d_4_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp#assignvariableop_22_conv1d_5_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp!assignvariableop_23_conv1d_5_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_24AssignVariableOp/assignvariableop_24_layer_normalization_6_gammaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp.assignvariableop_25_layer_normalization_6_betaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_26AssignVariableOp/assignvariableop_26_layer_normalization_7_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp.assignvariableop_27_layer_normalization_7_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp#assignvariableop_28_conv1d_6_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp!assignvariableop_29_conv1d_6_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp#assignvariableop_30_conv1d_7_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp!assignvariableop_31_conv1d_7_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp assignvariableop_32_dense_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOpassignvariableop_33_dense_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp"assignvariableop_34_dense_1_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp assignvariableop_35_dense_1_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_36AssignVariableOp5assignvariableop_36_multi_head_attention_query_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:€
AssignVariableOp_37AssignVariableOp3assignvariableop_37_multi_head_attention_query_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:€
AssignVariableOp_38AssignVariableOp3assignvariableop_38_multi_head_attention_key_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:’
AssignVariableOp_39AssignVariableOp1assignvariableop_39_multi_head_attention_key_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_40AssignVariableOp5assignvariableop_40_multi_head_attention_value_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:€
AssignVariableOp_41AssignVariableOp3assignvariableop_41_multi_head_attention_value_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_42AssignVariableOp@assignvariableop_42_multi_head_attention_attention_output_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:―
AssignVariableOp_43AssignVariableOp>assignvariableop_43_multi_head_attention_attention_output_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_44AssignVariableOp7assignvariableop_44_multi_head_attention_1_query_kernelIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_45AssignVariableOp5assignvariableop_45_multi_head_attention_1_query_biasIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_46AssignVariableOp5assignvariableop_46_multi_head_attention_1_key_kernelIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:€
AssignVariableOp_47AssignVariableOp3assignvariableop_47_multi_head_attention_1_key_biasIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_48AssignVariableOp7assignvariableop_48_multi_head_attention_1_value_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_49AssignVariableOp5assignvariableop_49_multi_head_attention_1_value_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_50AssignVariableOpBassignvariableop_50_multi_head_attention_1_attention_output_kernelIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_51AssignVariableOp@assignvariableop_51_multi_head_attention_1_attention_output_biasIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_52AssignVariableOp7assignvariableop_52_multi_head_attention_2_query_kernelIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_53AssignVariableOp5assignvariableop_53_multi_head_attention_2_query_biasIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_54AssignVariableOp5assignvariableop_54_multi_head_attention_2_key_kernelIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:€
AssignVariableOp_55AssignVariableOp3assignvariableop_55_multi_head_attention_2_key_biasIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_56AssignVariableOp7assignvariableop_56_multi_head_attention_2_value_kernelIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_57AssignVariableOp5assignvariableop_57_multi_head_attention_2_value_biasIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_58AssignVariableOpBassignvariableop_58_multi_head_attention_2_attention_output_kernelIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_59AssignVariableOp@assignvariableop_59_multi_head_attention_2_attention_output_biasIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_60AssignVariableOp7assignvariableop_60_multi_head_attention_3_query_kernelIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_61AssignVariableOp5assignvariableop_61_multi_head_attention_3_query_biasIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_62AssignVariableOp5assignvariableop_62_multi_head_attention_3_key_kernelIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:€
AssignVariableOp_63AssignVariableOp3assignvariableop_63_multi_head_attention_3_key_biasIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_64AssignVariableOp7assignvariableop_64_multi_head_attention_3_value_kernelIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_65AssignVariableOp5assignvariableop_65_multi_head_attention_3_value_biasIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_66AssignVariableOpBassignvariableop_66_multi_head_attention_3_attention_output_kernelIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_67AssignVariableOp@assignvariableop_67_multi_head_attention_3_attention_output_biasIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_68AssignVariableOpassignvariableop_68_iterIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_69AssignVariableOpassignvariableop_69_decayIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_70AssignVariableOp!assignvariableop_70_learning_rateIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_71AssignVariableOpassignvariableop_71_momentumIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_72AssignVariableOpassignvariableop_72_rhoIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_73AssignVariableOpassignvariableop_73_total_1Identity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_74AssignVariableOpassignvariableop_74_count_1Identity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_75AssignVariableOpassignvariableop_75_totalIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_76AssignVariableOpassignvariableop_76_countIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:’
AssignVariableOp_77AssignVariableOp1assignvariableop_77_layer_normalization_gamma_rmsIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:‘
AssignVariableOp_78AssignVariableOp0assignvariableop_78_layer_normalization_beta_rmsIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:€
AssignVariableOp_79AssignVariableOp3assignvariableop_79_layer_normalization_1_gamma_rmsIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_80AssignVariableOp2assignvariableop_80_layer_normalization_1_beta_rmsIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_81AssignVariableOp%assignvariableop_81_conv1d_kernel_rmsIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_82AssignVariableOp#assignvariableop_82_conv1d_bias_rmsIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_83AssignVariableOp'assignvariableop_83_conv1d_1_kernel_rmsIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_84AssignVariableOp%assignvariableop_84_conv1d_1_bias_rmsIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:€
AssignVariableOp_85AssignVariableOp3assignvariableop_85_layer_normalization_2_gamma_rmsIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_86AssignVariableOp2assignvariableop_86_layer_normalization_2_beta_rmsIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:€
AssignVariableOp_87AssignVariableOp3assignvariableop_87_layer_normalization_3_gamma_rmsIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_88AssignVariableOp2assignvariableop_88_layer_normalization_3_beta_rmsIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_89AssignVariableOp'assignvariableop_89_conv1d_2_kernel_rmsIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_90AssignVariableOp%assignvariableop_90_conv1d_2_bias_rmsIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_91AssignVariableOp'assignvariableop_91_conv1d_3_kernel_rmsIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_92AssignVariableOp%assignvariableop_92_conv1d_3_bias_rmsIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:€
AssignVariableOp_93AssignVariableOp3assignvariableop_93_layer_normalization_4_gamma_rmsIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_94AssignVariableOp2assignvariableop_94_layer_normalization_4_beta_rmsIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:€
AssignVariableOp_95AssignVariableOp3assignvariableop_95_layer_normalization_5_gamma_rmsIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_96AssignVariableOp2assignvariableop_96_layer_normalization_5_beta_rmsIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_97AssignVariableOp'assignvariableop_97_conv1d_4_kernel_rmsIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_98AssignVariableOp%assignvariableop_98_conv1d_4_bias_rmsIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_99AssignVariableOp'assignvariableop_99_conv1d_5_kernel_rmsIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_100AssignVariableOp&assignvariableop_100_conv1d_5_bias_rmsIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_101AssignVariableOp4assignvariableop_101_layer_normalization_6_gamma_rmsIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_102AssignVariableOp3assignvariableop_102_layer_normalization_6_beta_rmsIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_103AssignVariableOp4assignvariableop_103_layer_normalization_7_gamma_rmsIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_104AssignVariableOp3assignvariableop_104_layer_normalization_7_beta_rmsIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_105AssignVariableOp(assignvariableop_105_conv1d_6_kernel_rmsIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_106AssignVariableOp&assignvariableop_106_conv1d_6_bias_rmsIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_107AssignVariableOp(assignvariableop_107_conv1d_7_kernel_rmsIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_108AssignVariableOp&assignvariableop_108_conv1d_7_bias_rmsIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_109AssignVariableOp%assignvariableop_109_dense_kernel_rmsIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_110AssignVariableOp#assignvariableop_110_dense_bias_rmsIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_111AssignVariableOp'assignvariableop_111_dense_1_kernel_rmsIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_112AssignVariableOp%assignvariableop_112_dense_1_bias_rmsIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_113AssignVariableOp:assignvariableop_113_multi_head_attention_query_kernel_rmsIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_114AssignVariableOp8assignvariableop_114_multi_head_attention_query_bias_rmsIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_115AssignVariableOp8assignvariableop_115_multi_head_attention_key_kernel_rmsIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_116AssignVariableOp6assignvariableop_116_multi_head_attention_key_bias_rmsIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_117AssignVariableOp:assignvariableop_117_multi_head_attention_value_kernel_rmsIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_118AssignVariableOp8assignvariableop_118_multi_head_attention_value_bias_rmsIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:Έ
AssignVariableOp_119AssignVariableOpEassignvariableop_119_multi_head_attention_attention_output_kernel_rmsIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:Ά
AssignVariableOp_120AssignVariableOpCassignvariableop_120_multi_head_attention_attention_output_bias_rmsIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:―
AssignVariableOp_121AssignVariableOp<assignvariableop_121_multi_head_attention_1_query_kernel_rmsIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_122AssignVariableOp:assignvariableop_122_multi_head_attention_1_query_bias_rmsIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_123AssignVariableOp:assignvariableop_123_multi_head_attention_1_key_kernel_rmsIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_124AssignVariableOp8assignvariableop_124_multi_head_attention_1_key_bias_rmsIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:―
AssignVariableOp_125AssignVariableOp<assignvariableop_125_multi_head_attention_1_value_kernel_rmsIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_126AssignVariableOp:assignvariableop_126_multi_head_attention_1_value_bias_rmsIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:Ί
AssignVariableOp_127AssignVariableOpGassignvariableop_127_multi_head_attention_1_attention_output_kernel_rmsIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:Έ
AssignVariableOp_128AssignVariableOpEassignvariableop_128_multi_head_attention_1_attention_output_bias_rmsIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:―
AssignVariableOp_129AssignVariableOp<assignvariableop_129_multi_head_attention_2_query_kernel_rmsIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_130AssignVariableOp:assignvariableop_130_multi_head_attention_2_query_bias_rmsIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_131AssignVariableOp:assignvariableop_131_multi_head_attention_2_key_kernel_rmsIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_132AssignVariableOp8assignvariableop_132_multi_head_attention_2_key_bias_rmsIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:―
AssignVariableOp_133AssignVariableOp<assignvariableop_133_multi_head_attention_2_value_kernel_rmsIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_134AssignVariableOp:assignvariableop_134_multi_head_attention_2_value_bias_rmsIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:Ί
AssignVariableOp_135AssignVariableOpGassignvariableop_135_multi_head_attention_2_attention_output_kernel_rmsIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:Έ
AssignVariableOp_136AssignVariableOpEassignvariableop_136_multi_head_attention_2_attention_output_bias_rmsIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:―
AssignVariableOp_137AssignVariableOp<assignvariableop_137_multi_head_attention_3_query_kernel_rmsIdentity_137:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_138AssignVariableOp:assignvariableop_138_multi_head_attention_3_query_bias_rmsIdentity_138:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_139IdentityRestoreV2:tensors:139"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_139AssignVariableOp:assignvariableop_139_multi_head_attention_3_key_kernel_rmsIdentity_139:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_140IdentityRestoreV2:tensors:140"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_140AssignVariableOp8assignvariableop_140_multi_head_attention_3_key_bias_rmsIdentity_140:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_141IdentityRestoreV2:tensors:141"/device:CPU:0*
T0*
_output_shapes
:―
AssignVariableOp_141AssignVariableOp<assignvariableop_141_multi_head_attention_3_value_kernel_rmsIdentity_141:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_142IdentityRestoreV2:tensors:142"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_142AssignVariableOp:assignvariableop_142_multi_head_attention_3_value_bias_rmsIdentity_142:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_143IdentityRestoreV2:tensors:143"/device:CPU:0*
T0*
_output_shapes
:Ί
AssignVariableOp_143AssignVariableOpGassignvariableop_143_multi_head_attention_3_attention_output_kernel_rmsIdentity_143:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_144IdentityRestoreV2:tensors:144"/device:CPU:0*
T0*
_output_shapes
:Έ
AssignVariableOp_144AssignVariableOpEassignvariableop_144_multi_head_attention_3_attention_output_bias_rmsIdentity_144:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 σ
Identity_145Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_146IdentityIdentity_145:output:0^NoOp_1*
T0*
_output_shapes
: ί
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_146Identity_146:output:0*Ή
_input_shapes§
€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322,
AssignVariableOp_133AssignVariableOp_1332,
AssignVariableOp_134AssignVariableOp_1342,
AssignVariableOp_135AssignVariableOp_1352,
AssignVariableOp_136AssignVariableOp_1362,
AssignVariableOp_137AssignVariableOp_1372,
AssignVariableOp_138AssignVariableOp_1382,
AssignVariableOp_139AssignVariableOp_1392*
AssignVariableOp_14AssignVariableOp_142,
AssignVariableOp_140AssignVariableOp_1402,
AssignVariableOp_141AssignVariableOp_1412,
AssignVariableOp_142AssignVariableOp_1422,
AssignVariableOp_143AssignVariableOp_1432,
AssignVariableOp_144AssignVariableOp_1442*
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
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix


c
D__inference_dropout_4_layer_call_and_return_conditional_losses_18098

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:?????????ΘC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:?????????Θ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????Θt
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????Θn
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:?????????Θ^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:?????????Θ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs

b
)__inference_dropout_5_layer_call_fn_18164

inputs
identity’StatefulPartitionedCallΔ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_14589t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
ύ

P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_18129

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(r
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:?????????Θ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:?????????Θl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¬
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:?????????Θ*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:?????????Θb
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:?????????Θ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θh
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θw
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Θv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????Θw
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????Θg
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:?????????Θ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
₯±
ά
@__inference_model_layer_call_and_return_conditional_losses_15861
input_1'
layer_normalization_15684:'
layer_normalization_15686:0
multi_head_attention_15689:2,
multi_head_attention_15691:20
multi_head_attention_15693:2,
multi_head_attention_15695:20
multi_head_attention_15697:2,
multi_head_attention_15699:20
multi_head_attention_15701:2(
multi_head_attention_15703:)
layer_normalization_1_15708:)
layer_normalization_1_15710:"
conv1d_15713:
conv1d_15715:$
conv1d_1_15719:
conv1d_1_15721:)
layer_normalization_2_15725:)
layer_normalization_2_15727:2
multi_head_attention_1_15730:2.
multi_head_attention_1_15732:22
multi_head_attention_1_15734:2.
multi_head_attention_1_15736:22
multi_head_attention_1_15738:2.
multi_head_attention_1_15740:22
multi_head_attention_1_15742:2*
multi_head_attention_1_15744:)
layer_normalization_3_15749:)
layer_normalization_3_15751:$
conv1d_2_15754:
conv1d_2_15756:$
conv1d_3_15760:
conv1d_3_15762:)
layer_normalization_4_15766:)
layer_normalization_4_15768:2
multi_head_attention_2_15771:2.
multi_head_attention_2_15773:22
multi_head_attention_2_15775:2.
multi_head_attention_2_15777:22
multi_head_attention_2_15779:2.
multi_head_attention_2_15781:22
multi_head_attention_2_15783:2*
multi_head_attention_2_15785:)
layer_normalization_5_15790:)
layer_normalization_5_15792:$
conv1d_4_15795:
conv1d_4_15797:$
conv1d_5_15801:
conv1d_5_15803:)
layer_normalization_6_15807:)
layer_normalization_6_15809:2
multi_head_attention_3_15812:2.
multi_head_attention_3_15814:22
multi_head_attention_3_15816:2.
multi_head_attention_3_15818:22
multi_head_attention_3_15820:2.
multi_head_attention_3_15822:22
multi_head_attention_3_15824:2*
multi_head_attention_3_15826:)
layer_normalization_7_15831:)
layer_normalization_7_15833:$
conv1d_6_15836:
conv1d_6_15838:$
conv1d_7_15842:
conv1d_7_15844:
dense_15849:
Θ
dense_15851:	 
dense_1_15855:	
dense_1_15857:
identity’conv1d/StatefulPartitionedCall’ conv1d_1/StatefulPartitionedCall’ conv1d_2/StatefulPartitionedCall’ conv1d_3/StatefulPartitionedCall’ conv1d_4/StatefulPartitionedCall’ conv1d_5/StatefulPartitionedCall’ conv1d_6/StatefulPartitionedCall’ conv1d_7/StatefulPartitionedCall’dense/StatefulPartitionedCall’dense_1/StatefulPartitionedCall’+layer_normalization/StatefulPartitionedCall’-layer_normalization_1/StatefulPartitionedCall’-layer_normalization_2/StatefulPartitionedCall’-layer_normalization_3/StatefulPartitionedCall’-layer_normalization_4/StatefulPartitionedCall’-layer_normalization_5/StatefulPartitionedCall’-layer_normalization_6/StatefulPartitionedCall’-layer_normalization_7/StatefulPartitionedCall’,multi_head_attention/StatefulPartitionedCall’.multi_head_attention_1/StatefulPartitionedCall’.multi_head_attention_2/StatefulPartitionedCall’.multi_head_attention_3/StatefulPartitionedCall
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCallinput_1layer_normalization_15684layer_normalization_15686*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_13512»
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:04layer_normalization/StatefulPartitionedCall:output:0multi_head_attention_15689multi_head_attention_15691multi_head_attention_15693multi_head_attention_15695multi_head_attention_15697multi_head_attention_15699multi_head_attention_15701multi_head_attention_15703*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_13553ι
dropout/PartitionedCallPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_13576
tf.__operators__.add/AddV2AddV2 dropout/PartitionedCall:output:0input_1*
T0*,
_output_shapes
:?????????ΘΎ
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0layer_normalization_1_15708layer_normalization_1_15710*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_13601
conv1d/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0conv1d_15713conv1d_15715*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_13623ί
dropout_1/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_13634
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conv1d_1_15719conv1d_1_15721*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_13651§
tf.__operators__.add_1/AddV2AddV2)conv1d_1/StatefulPartitionedCall:output:0tf.__operators__.add/AddV2:z:0*
T0*,
_output_shapes
:?????????Θΐ
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0layer_normalization_2_15725layer_normalization_2_15727*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_13680Σ
.multi_head_attention_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:06layer_normalization_2/StatefulPartitionedCall:output:0multi_head_attention_1_15730multi_head_attention_1_15732multi_head_attention_1_15734multi_head_attention_1_15736multi_head_attention_1_15738multi_head_attention_1_15740multi_head_attention_1_15742multi_head_attention_1_15744*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_13721ο
dropout_2/PartitionedCallPartitionedCall7multi_head_attention_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_13744’
tf.__operators__.add_2/AddV2AddV2"dropout_2/PartitionedCall:output:0 tf.__operators__.add_1/AddV2:z:0*
T0*,
_output_shapes
:?????????Θΐ
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0layer_normalization_3_15749layer_normalization_3_15751*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_13769’
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_3/StatefulPartitionedCall:output:0conv1d_2_15754conv1d_2_15756*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_13791α
dropout_3/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_13802
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0conv1d_3_15760conv1d_3_15762*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_3_layer_call_and_return_conditional_losses_13819©
tf.__operators__.add_3/AddV2AddV2)conv1d_3/StatefulPartitionedCall:output:0 tf.__operators__.add_2/AddV2:z:0*
T0*,
_output_shapes
:?????????Θΐ
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_3/AddV2:z:0layer_normalization_4_15766layer_normalization_4_15768*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_13848Σ
.multi_head_attention_2/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:06layer_normalization_4/StatefulPartitionedCall:output:0multi_head_attention_2_15771multi_head_attention_2_15773multi_head_attention_2_15775multi_head_attention_2_15777multi_head_attention_2_15779multi_head_attention_2_15781multi_head_attention_2_15783multi_head_attention_2_15785*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_13889ο
dropout_4/PartitionedCallPartitionedCall7multi_head_attention_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_13912’
tf.__operators__.add_4/AddV2AddV2"dropout_4/PartitionedCall:output:0 tf.__operators__.add_3/AddV2:z:0*
T0*,
_output_shapes
:?????????Θΐ
-layer_normalization_5/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_4/AddV2:z:0layer_normalization_5_15790layer_normalization_5_15792*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_13937’
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_5/StatefulPartitionedCall:output:0conv1d_4_15795conv1d_4_15797*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_4_layer_call_and_return_conditional_losses_13959α
dropout_5/PartitionedCallPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_13970
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0conv1d_5_15801conv1d_5_15803*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_5_layer_call_and_return_conditional_losses_13987©
tf.__operators__.add_5/AddV2AddV2)conv1d_5/StatefulPartitionedCall:output:0 tf.__operators__.add_4/AddV2:z:0*
T0*,
_output_shapes
:?????????Θΐ
-layer_normalization_6/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_5/AddV2:z:0layer_normalization_6_15807layer_normalization_6_15809*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_layer_normalization_6_layer_call_and_return_conditional_losses_14016Σ
.multi_head_attention_3/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_6/StatefulPartitionedCall:output:06layer_normalization_6/StatefulPartitionedCall:output:0multi_head_attention_3_15812multi_head_attention_3_15814multi_head_attention_3_15816multi_head_attention_3_15818multi_head_attention_3_15820multi_head_attention_3_15822multi_head_attention_3_15824multi_head_attention_3_15826*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_multi_head_attention_3_layer_call_and_return_conditional_losses_14057ο
dropout_6/PartitionedCallPartitionedCall7multi_head_attention_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_14080’
tf.__operators__.add_6/AddV2AddV2"dropout_6/PartitionedCall:output:0 tf.__operators__.add_5/AddV2:z:0*
T0*,
_output_shapes
:?????????Θΐ
-layer_normalization_7/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_6/AddV2:z:0layer_normalization_7_15831layer_normalization_7_15833*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_layer_normalization_7_layer_call_and_return_conditional_losses_14105’
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_7/StatefulPartitionedCall:output:0conv1d_6_15836conv1d_6_15838*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_6_layer_call_and_return_conditional_losses_14127α
dropout_7/PartitionedCallPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_14138
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0conv1d_7_15842conv1d_7_15844*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_7_layer_call_and_return_conditional_losses_14155©
tf.__operators__.add_7/AddV2AddV2)conv1d_7/StatefulPartitionedCall:output:0 tf.__operators__.add_6/AddV2:z:0*
T0*,
_output_shapes
:?????????Θς
(global_average_pooling1d/PartitionedCallPartitionedCall tf.__operators__.add_7/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_13480
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_15849dense_15851*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_14174Ϊ
dropout_8/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_14185
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0dense_1_15855dense_1_15857*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_14197w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????ή
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall.^layer_normalization_3/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall.^layer_normalization_5/StatefulPartitionedCall.^layer_normalization_6/StatefulPartitionedCall.^layer_normalization_7/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall/^multi_head_attention_1/StatefulPartitionedCall/^multi_head_attention_2/StatefulPartitionedCall/^multi_head_attention_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*΅
_input_shapes£
 :?????????Θ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2^
-layer_normalization_3/StatefulPartitionedCall-layer_normalization_3/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall2^
-layer_normalization_5/StatefulPartitionedCall-layer_normalization_5/StatefulPartitionedCall2^
-layer_normalization_6/StatefulPartitionedCall-layer_normalization_6/StatefulPartitionedCall2^
-layer_normalization_7/StatefulPartitionedCall-layer_normalization_7/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall2`
.multi_head_attention_1/StatefulPartitionedCall.multi_head_attention_1/StatefulPartitionedCall2`
.multi_head_attention_2/StatefulPartitionedCall.multi_head_attention_2/StatefulPartitionedCall2`
.multi_head_attention_3/StatefulPartitionedCall.multi_head_attention_3/StatefulPartitionedCall:U Q
,
_output_shapes
:?????????Θ
!
_user_specified_name	input_1
?
`
'__inference_dropout_layer_call_fn_17509

inputs
identity’StatefulPartitionedCallΒ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_14978t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
£

τ
@__inference_dense_layer_call_and_return_conditional_losses_14174

inputs2
matmul_readvariableop_resource:
Θ.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Θ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
λ
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_18086

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:?????????Θ`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:?????????Θ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
Α

'__inference_dense_1_layer_call_fn_18558

inputs
unknown:	
	unknown_0:
identity’StatefulPartitionedCallΧ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_14197o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ζ

3__inference_layer_normalization_layer_call_fn_17356

inputs
unknown:
	unknown_0:
identity’StatefulPartitionedCallθ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_13512t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
φ

C__inference_conv1d_3_layer_call_and_return_conditional_losses_13819

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θd
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs


c
D__inference_dropout_4_layer_call_and_return_conditional_losses_14632

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:?????????ΘC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:?????????Θ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????Θt
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????Θn
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:?????????Θ^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:?????????Θ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs


a
B__inference_dropout_layer_call_and_return_conditional_losses_17526

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:?????????ΘC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:?????????Θ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????Θt
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????Θn
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:?????????Θ^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:?????????Θ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
±
E
)__inference_dropout_7_layer_call_fn_18445

inputs
identity΄
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_14138e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:?????????Θ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
φ

C__inference_conv1d_1_layer_call_and_return_conditional_losses_17633

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Θ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????Θ*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:?????????Θ*
squeeze_dims

ύ????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θd
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs

b
)__inference_dropout_3_layer_call_fn_17878

inputs
identity’StatefulPartitionedCallΔ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_14762t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
·2
ϊ
Q__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_17785	
query	
valueA
+query_einsum_einsum_readvariableop_resource:23
!query_add_readvariableop_resource:2?
)key_einsum_einsum_readvariableop_resource:21
key_add_readvariableop_resource:2A
+value_einsum_einsum_readvariableop_resource:23
!value_add_readvariableop_resource:2L
6attention_output_einsum_einsum_readvariableop_resource:2:
,attention_output_add_readvariableop_resource:
identity’#attention_output/add/ReadVariableOp’-attention_output/einsum/Einsum/ReadVariableOp’key/add/ReadVariableOp’ key/einsum/Einsum/ReadVariableOp’query/add/ReadVariableOp’"query/einsum/Einsum/ReadVariableOp’value/add/ReadVariableOp’"value/einsum/Einsum/ReadVariableOp
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:2*
dtype0
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0­
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:2*
dtype0
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:2*
dtype0
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΓΠ>d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:?????????Θ2
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:?????????ΘΘ*
equationaecd,abcd->acbep
softmax_1/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:?????????ΘΘ]
dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ?
dropout_10/dropout/MulMulsoftmax_1/Softmax:softmax:0!dropout_10/dropout/Const:output:0*
T0*1
_output_shapes
:?????????ΘΘc
dropout_10/dropout/ShapeShapesoftmax_1/Softmax:softmax:0*
T0*
_output_shapes
:¬
/dropout_10/dropout/random_uniform/RandomUniformRandomUniform!dropout_10/dropout/Shape:output:0*
T0*1
_output_shapes
:?????????ΘΘ*
dtype0f
!dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >Ρ
dropout_10/dropout/GreaterEqualGreaterEqual8dropout_10/dropout/random_uniform/RandomUniform:output:0*dropout_10/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:?????????ΘΘ
dropout_10/dropout/CastCast#dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:?????????ΘΘ
dropout_10/dropout/Mul_1Muldropout_10/dropout/Mul:z:0dropout_10/dropout/Cast:y:0*
T0*1
_output_shapes
:?????????ΘΘ©
einsum_1/EinsumEinsumdropout_10/dropout/Mul_1:z:0value/add:z:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationacbe,aecd->abcd¨
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0Φ
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????Θ*
equationabcd,cde->abe
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0ͺ
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θl
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:?????????ΘΨ
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Θ:?????????Θ: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:?????????Θ

_user_specified_namequery:SO
,
_output_shapes
:?????????Θ

_user_specified_namevalue

b
)__inference_dropout_2_layer_call_fn_17795

inputs
identity’StatefulPartitionedCallΔ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_14805t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
Ψ

(__inference_conv1d_4_layer_call_fn_18138

inputs
unknown:
	unknown_0:
identity’StatefulPartitionedCallέ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_4_layer_call_and_return_conditional_losses_13959t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
±
E
)__inference_dropout_6_layer_call_fn_18362

inputs
identity΄
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_14080e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:?????????Θ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs

b
)__inference_dropout_7_layer_call_fn_18450

inputs
identity’StatefulPartitionedCallΔ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_14416t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
κ

5__inference_layer_normalization_1_layer_call_fn_17535

inputs
unknown:
	unknown_0:
identity’StatefulPartitionedCallκ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_13601t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
λ
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_13912

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:?????????Θ`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:?????????Θ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
λ
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_13744

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:?????????Θ`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:?????????Θ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
κ

5__inference_layer_normalization_2_layer_call_fn_17642

inputs
unknown:
	unknown_0:
identity’StatefulPartitionedCallκ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_13680t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs


c
D__inference_dropout_3_layer_call_and_return_conditional_losses_17895

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:?????????ΘC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:?????????Θ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????Θt
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????Θn
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:?????????Θ^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:?????????Θ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
Ψ

(__inference_conv1d_1_layer_call_fn_17618

inputs
unknown:
	unknown_0:
identity’StatefulPartitionedCallέ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_13651t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
κ

5__inference_layer_normalization_7_layer_call_fn_18393

inputs
unknown:
	unknown_0:
identity’StatefulPartitionedCallκ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_layer_normalization_7_layer_call_and_return_conditional_losses_14105t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
‘
E
)__inference_dropout_8_layer_call_fn_18527

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_14185a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
Ώ
!
@__inference_model_layer_call_and_return_conditional_losses_15401

inputs'
layer_normalization_15224:'
layer_normalization_15226:0
multi_head_attention_15229:2,
multi_head_attention_15231:20
multi_head_attention_15233:2,
multi_head_attention_15235:20
multi_head_attention_15237:2,
multi_head_attention_15239:20
multi_head_attention_15241:2(
multi_head_attention_15243:)
layer_normalization_1_15248:)
layer_normalization_1_15250:"
conv1d_15253:
conv1d_15255:$
conv1d_1_15259:
conv1d_1_15261:)
layer_normalization_2_15265:)
layer_normalization_2_15267:2
multi_head_attention_1_15270:2.
multi_head_attention_1_15272:22
multi_head_attention_1_15274:2.
multi_head_attention_1_15276:22
multi_head_attention_1_15278:2.
multi_head_attention_1_15280:22
multi_head_attention_1_15282:2*
multi_head_attention_1_15284:)
layer_normalization_3_15289:)
layer_normalization_3_15291:$
conv1d_2_15294:
conv1d_2_15296:$
conv1d_3_15300:
conv1d_3_15302:)
layer_normalization_4_15306:)
layer_normalization_4_15308:2
multi_head_attention_2_15311:2.
multi_head_attention_2_15313:22
multi_head_attention_2_15315:2.
multi_head_attention_2_15317:22
multi_head_attention_2_15319:2.
multi_head_attention_2_15321:22
multi_head_attention_2_15323:2*
multi_head_attention_2_15325:)
layer_normalization_5_15330:)
layer_normalization_5_15332:$
conv1d_4_15335:
conv1d_4_15337:$
conv1d_5_15341:
conv1d_5_15343:)
layer_normalization_6_15347:)
layer_normalization_6_15349:2
multi_head_attention_3_15352:2.
multi_head_attention_3_15354:22
multi_head_attention_3_15356:2.
multi_head_attention_3_15358:22
multi_head_attention_3_15360:2.
multi_head_attention_3_15362:22
multi_head_attention_3_15364:2*
multi_head_attention_3_15366:)
layer_normalization_7_15371:)
layer_normalization_7_15373:$
conv1d_6_15376:
conv1d_6_15378:$
conv1d_7_15382:
conv1d_7_15384:
dense_15389:
Θ
dense_15391:	 
dense_1_15395:	
dense_1_15397:
identity’conv1d/StatefulPartitionedCall’ conv1d_1/StatefulPartitionedCall’ conv1d_2/StatefulPartitionedCall’ conv1d_3/StatefulPartitionedCall’ conv1d_4/StatefulPartitionedCall’ conv1d_5/StatefulPartitionedCall’ conv1d_6/StatefulPartitionedCall’ conv1d_7/StatefulPartitionedCall’dense/StatefulPartitionedCall’dense_1/StatefulPartitionedCall’dropout/StatefulPartitionedCall’!dropout_1/StatefulPartitionedCall’!dropout_2/StatefulPartitionedCall’!dropout_3/StatefulPartitionedCall’!dropout_4/StatefulPartitionedCall’!dropout_5/StatefulPartitionedCall’!dropout_6/StatefulPartitionedCall’!dropout_7/StatefulPartitionedCall’!dropout_8/StatefulPartitionedCall’+layer_normalization/StatefulPartitionedCall’-layer_normalization_1/StatefulPartitionedCall’-layer_normalization_2/StatefulPartitionedCall’-layer_normalization_3/StatefulPartitionedCall’-layer_normalization_4/StatefulPartitionedCall’-layer_normalization_5/StatefulPartitionedCall’-layer_normalization_6/StatefulPartitionedCall’-layer_normalization_7/StatefulPartitionedCall’,multi_head_attention/StatefulPartitionedCall’.multi_head_attention_1/StatefulPartitionedCall’.multi_head_attention_2/StatefulPartitionedCall’.multi_head_attention_3/StatefulPartitionedCall
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCallinputslayer_normalization_15224layer_normalization_15226*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_13512»
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:04layer_normalization/StatefulPartitionedCall:output:0multi_head_attention_15229multi_head_attention_15231multi_head_attention_15233multi_head_attention_15235multi_head_attention_15237multi_head_attention_15239multi_head_attention_15241multi_head_attention_15243*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_15049ω
dropout/StatefulPartitionedCallStatefulPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_14978
tf.__operators__.add/AddV2AddV2(dropout/StatefulPartitionedCall:output:0inputs*
T0*,
_output_shapes
:?????????ΘΎ
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0layer_normalization_1_15248layer_normalization_1_15250*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_13601
conv1d/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0conv1d_15253conv1d_15255*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_13623
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_14935
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv1d_1_15259conv1d_1_15261*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_13651§
tf.__operators__.add_1/AddV2AddV2)conv1d_1/StatefulPartitionedCall:output:0tf.__operators__.add/AddV2:z:0*
T0*,
_output_shapes
:?????????Θΐ
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0layer_normalization_2_15265layer_normalization_2_15267*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_13680Σ
.multi_head_attention_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:06layer_normalization_2/StatefulPartitionedCall:output:0multi_head_attention_1_15270multi_head_attention_1_15272multi_head_attention_1_15274multi_head_attention_1_15276multi_head_attention_1_15278multi_head_attention_1_15280multi_head_attention_1_15282multi_head_attention_1_15284*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_14876£
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall7multi_head_attention_1/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_14805ͺ
tf.__operators__.add_2/AddV2AddV2*dropout_2/StatefulPartitionedCall:output:0 tf.__operators__.add_1/AddV2:z:0*
T0*,
_output_shapes
:?????????Θΐ
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0layer_normalization_3_15289layer_normalization_3_15291*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_13769’
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_3/StatefulPartitionedCall:output:0conv1d_2_15294conv1d_2_15296*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_13791
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_14762
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0conv1d_3_15300conv1d_3_15302*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_3_layer_call_and_return_conditional_losses_13819©
tf.__operators__.add_3/AddV2AddV2)conv1d_3/StatefulPartitionedCall:output:0 tf.__operators__.add_2/AddV2:z:0*
T0*,
_output_shapes
:?????????Θΐ
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_3/AddV2:z:0layer_normalization_4_15306layer_normalization_4_15308*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_13848Σ
.multi_head_attention_2/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:06layer_normalization_4/StatefulPartitionedCall:output:0multi_head_attention_2_15311multi_head_attention_2_15313multi_head_attention_2_15315multi_head_attention_2_15317multi_head_attention_2_15319multi_head_attention_2_15321multi_head_attention_2_15323multi_head_attention_2_15325*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_14703£
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall7multi_head_attention_2/StatefulPartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_14632ͺ
tf.__operators__.add_4/AddV2AddV2*dropout_4/StatefulPartitionedCall:output:0 tf.__operators__.add_3/AddV2:z:0*
T0*,
_output_shapes
:?????????Θΐ
-layer_normalization_5/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_4/AddV2:z:0layer_normalization_5_15330layer_normalization_5_15332*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_13937’
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_5/StatefulPartitionedCall:output:0conv1d_4_15335conv1d_4_15337*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_4_layer_call_and_return_conditional_losses_13959
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_14589
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0conv1d_5_15341conv1d_5_15343*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_5_layer_call_and_return_conditional_losses_13987©
tf.__operators__.add_5/AddV2AddV2)conv1d_5/StatefulPartitionedCall:output:0 tf.__operators__.add_4/AddV2:z:0*
T0*,
_output_shapes
:?????????Θΐ
-layer_normalization_6/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_5/AddV2:z:0layer_normalization_6_15347layer_normalization_6_15349*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_layer_normalization_6_layer_call_and_return_conditional_losses_14016Σ
.multi_head_attention_3/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_6/StatefulPartitionedCall:output:06layer_normalization_6/StatefulPartitionedCall:output:0multi_head_attention_3_15352multi_head_attention_3_15354multi_head_attention_3_15356multi_head_attention_3_15358multi_head_attention_3_15360multi_head_attention_3_15362multi_head_attention_3_15364multi_head_attention_3_15366*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_multi_head_attention_3_layer_call_and_return_conditional_losses_14530£
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall7multi_head_attention_3/StatefulPartitionedCall:output:0"^dropout_5/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_14459ͺ
tf.__operators__.add_6/AddV2AddV2*dropout_6/StatefulPartitionedCall:output:0 tf.__operators__.add_5/AddV2:z:0*
T0*,
_output_shapes
:?????????Θΐ
-layer_normalization_7/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_6/AddV2:z:0layer_normalization_7_15371layer_normalization_7_15373*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_layer_normalization_7_layer_call_and_return_conditional_losses_14105’
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_7/StatefulPartitionedCall:output:0conv1d_6_15376conv1d_6_15378*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_6_layer_call_and_return_conditional_losses_14127
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_14416
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0conv1d_7_15382conv1d_7_15384*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_7_layer_call_and_return_conditional_losses_14155©
tf.__operators__.add_7/AddV2AddV2)conv1d_7/StatefulPartitionedCall:output:0 tf.__operators__.add_6/AddV2:z:0*
T0*,
_output_shapes
:?????????Θς
(global_average_pooling1d/PartitionedCallPartitionedCall tf.__operators__.add_7/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_13480
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_15389dense_15391*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_14174
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_14373
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0dense_1_15395dense_1_15397*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_14197w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 

NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall.^layer_normalization_3/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall.^layer_normalization_5/StatefulPartitionedCall.^layer_normalization_6/StatefulPartitionedCall.^layer_normalization_7/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall/^multi_head_attention_1/StatefulPartitionedCall/^multi_head_attention_2/StatefulPartitionedCall/^multi_head_attention_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*΅
_input_shapes£
 :?????????Θ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2^
-layer_normalization_3/StatefulPartitionedCall-layer_normalization_3/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall2^
-layer_normalization_5/StatefulPartitionedCall-layer_normalization_5/StatefulPartitionedCall2^
-layer_normalization_6/StatefulPartitionedCall-layer_normalization_6/StatefulPartitionedCall2^
-layer_normalization_7/StatefulPartitionedCall-layer_normalization_7/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall2`
.multi_head_attention_1/StatefulPartitionedCall.multi_head_attention_1/StatefulPartitionedCall2`
.multi_head_attention_2/StatefulPartitionedCall.multi_head_attention_2/StatefulPartitionedCall2`
.multi_head_attention_3/StatefulPartitionedCall.multi_head_attention_3/StatefulPartitionedCall:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
΅*
ψ
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_13553	
query	
valueA
+query_einsum_einsum_readvariableop_resource:23
!query_add_readvariableop_resource:2?
)key_einsum_einsum_readvariableop_resource:21
key_add_readvariableop_resource:2A
+value_einsum_einsum_readvariableop_resource:23
!value_add_readvariableop_resource:2L
6attention_output_einsum_einsum_readvariableop_resource:2:
,attention_output_add_readvariableop_resource:
identity’#attention_output/add/ReadVariableOp’-attention_output/einsum/Einsum/ReadVariableOp’key/add/ReadVariableOp’ key/einsum/Einsum/ReadVariableOp’query/add/ReadVariableOp’"query/einsum/Einsum/ReadVariableOp’value/add/ReadVariableOp’"value/einsum/Einsum/ReadVariableOp
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:2*
dtype0
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0­
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:2*
dtype0
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:2*
dtype0
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΓΠ>d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:?????????Θ2
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:?????????ΘΘ*
equationaecd,abcd->acben
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:?????????ΘΘu
dropout_9/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*1
_output_shapes
:?????????ΘΘ¨
einsum_1/EinsumEinsumdropout_9/Identity:output:0value/add:z:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationacbe,aecd->abcd¨
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0Φ
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????Θ*
equationabcd,cde->abe
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0ͺ
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θl
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:?????????ΘΨ
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Θ:?????????Θ: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:?????????Θ

_user_specified_namequery:SO
,
_output_shapes
:?????????Θ

_user_specified_namevalue
λ
b
D__inference_dropout_7_layer_call_and_return_conditional_losses_18455

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:?????????Θ`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:?????????Θ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
±
E
)__inference_dropout_1_layer_call_fn_17587

inputs
identity΄
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_13634e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:?????????Θ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
λ
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_17597

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:?????????Θ`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:?????????Θ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
·2
ϊ
Q__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_18071	
query	
valueA
+query_einsum_einsum_readvariableop_resource:23
!query_add_readvariableop_resource:2?
)key_einsum_einsum_readvariableop_resource:21
key_add_readvariableop_resource:2A
+value_einsum_einsum_readvariableop_resource:23
!value_add_readvariableop_resource:2L
6attention_output_einsum_einsum_readvariableop_resource:2:
,attention_output_add_readvariableop_resource:
identity’#attention_output/add/ReadVariableOp’-attention_output/einsum/Einsum/ReadVariableOp’key/add/ReadVariableOp’ key/einsum/Einsum/ReadVariableOp’query/add/ReadVariableOp’"query/einsum/Einsum/ReadVariableOp’value/add/ReadVariableOp’"value/einsum/Einsum/ReadVariableOp
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:2*
dtype0
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0­
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:2*
dtype0
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:2*
dtype0
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Θ2J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΓΠ>d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:?????????Θ2
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:?????????ΘΘ*
equationaecd,abcd->acbep
softmax_2/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:?????????ΘΘ]
dropout_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ?
dropout_11/dropout/MulMulsoftmax_2/Softmax:softmax:0!dropout_11/dropout/Const:output:0*
T0*1
_output_shapes
:?????????ΘΘc
dropout_11/dropout/ShapeShapesoftmax_2/Softmax:softmax:0*
T0*
_output_shapes
:¬
/dropout_11/dropout/random_uniform/RandomUniformRandomUniform!dropout_11/dropout/Shape:output:0*
T0*1
_output_shapes
:?????????ΘΘ*
dtype0f
!dropout_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >Ρ
dropout_11/dropout/GreaterEqualGreaterEqual8dropout_11/dropout/random_uniform/RandomUniform:output:0*dropout_11/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:?????????ΘΘ
dropout_11/dropout/CastCast#dropout_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:?????????ΘΘ
dropout_11/dropout/Mul_1Muldropout_11/dropout/Mul:z:0dropout_11/dropout/Cast:y:0*
T0*1
_output_shapes
:?????????ΘΘ©
einsum_1/EinsumEinsumdropout_11/dropout/Mul_1:z:0value/add:z:0*
N*
T0*0
_output_shapes
:?????????Θ2*
equationacbe,aecd->abcd¨
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:2*
dtype0Φ
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????Θ*
equationabcd,cde->abe
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0ͺ
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????Θl
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:?????????ΘΨ
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Θ:?????????Θ: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:?????????Θ

_user_specified_namequery:SO
,
_output_shapes
:?????????Θ

_user_specified_namevalue

ι
4__inference_multi_head_attention_layer_call_fn_17422	
query	
value
unknown:2
	unknown_0:2
	unknown_1:2
	unknown_2:2
	unknown_3:2
	unknown_4:2
	unknown_5:2
	unknown_6:
identity’StatefulPartitionedCallΎ
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_15049t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:?????????Θ:?????????Θ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:?????????Θ

_user_specified_namequery:SO
,
_output_shapes
:?????????Θ

_user_specified_namevalue
Ψ

(__inference_conv1d_5_layer_call_fn_18190

inputs
unknown:
	unknown_0:
identity’StatefulPartitionedCallέ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????Θ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_5_layer_call_and_return_conditional_losses_13987t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????Θ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????Θ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs
λ
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_13634

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:?????????Θ`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:?????????Θ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs


c
D__inference_dropout_6_layer_call_and_return_conditional_losses_18384

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:?????????ΘC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:?????????Θ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????Θt
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????Θn
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:?????????Θ^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:?????????Θ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs


c
D__inference_dropout_7_layer_call_and_return_conditional_losses_18467

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ͺͺ?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:?????????ΘC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:?????????Θ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????Θt
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????Θn
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:?????????Θ^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:?????????Θ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????Θ:T P
,
_output_shapes
:?????????Θ
 
_user_specified_nameinputs"ΏL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*―
serving_default
@
input_15
serving_default_input_1:0?????????Θ;
dense_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:―
ί

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer-12
layer-13
layer_with_weights-7
layer-14
layer_with_weights-8
layer-15
layer-16
layer_with_weights-9
layer-17
layer-18
layer_with_weights-10
layer-19
layer_with_weights-11
layer-20
layer-21
layer-22
layer_with_weights-12
layer-23
layer_with_weights-13
layer-24
layer-25
layer_with_weights-14
layer-26
layer-27
layer_with_weights-15
layer-28
layer_with_weights-16
layer-29
layer-30
 layer-31
!layer_with_weights-17
!layer-32
"layer_with_weights-18
"layer-33
#layer-34
$layer_with_weights-19
$layer-35
%layer-36
&layer-37
'layer_with_weights-20
'layer-38
(layer-39
)layer_with_weights-21
)layer-40
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
0_default_save_signature
1	optimizer
2
signatures"
_tf_keras_network
"
_tf_keras_input_layer
Δ
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses
9axis
	:gamma
;beta"
_tf_keras_layer
£
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses
B_attention_axes
C_query_dense
D
_key_dense
E_value_dense
F_softmax
G_dropout_layer
H_output_dense"
_tf_keras_layer
Ό
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses
O_random_generator"
_tf_keras_layer
(
P	keras_api"
_tf_keras_layer
Δ
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses
Waxis
	Xgamma
Ybeta"
_tf_keras_layer
έ
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses

`kernel
abias
 b_jit_compiled_convolution_op"
_tf_keras_layer
Ό
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses
i_random_generator"
_tf_keras_layer
έ
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses

pkernel
qbias
 r_jit_compiled_convolution_op"
_tf_keras_layer
(
s	keras_api"
_tf_keras_layer
Δ
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses
zaxis
	{gamma
|beta"
_tf_keras_layer
­
}	variables
~trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_attention_axes
_query_dense

_key_dense
_value_dense
_softmax
_dropout_layer
_output_dense"
_tf_keras_layer
Γ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator"
_tf_keras_layer
)
	keras_api"
_tf_keras_layer
Ν
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis

gamma
	beta"
_tf_keras_layer
ζ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses
‘kernel
	’bias
!£_jit_compiled_convolution_op"
_tf_keras_layer
Γ
€	variables
₯trainable_variables
¦regularization_losses
§	keras_api
¨__call__
+©&call_and_return_all_conditional_losses
ͺ_random_generator"
_tf_keras_layer
ζ
«	variables
¬trainable_variables
­regularization_losses
?	keras_api
―__call__
+°&call_and_return_all_conditional_losses
±kernel
	²bias
!³_jit_compiled_convolution_op"
_tf_keras_layer
)
΄	keras_api"
_tf_keras_layer
Ν
΅	variables
Άtrainable_variables
·regularization_losses
Έ	keras_api
Ή__call__
+Ί&call_and_return_all_conditional_losses
	»axis

Όgamma
	½beta"
_tf_keras_layer
°
Ύ	variables
Ώtrainable_variables
ΐregularization_losses
Α	keras_api
Β__call__
+Γ&call_and_return_all_conditional_losses
Δ_attention_axes
Ε_query_dense
Ζ
_key_dense
Η_value_dense
Θ_softmax
Ι_dropout_layer
Κ_output_dense"
_tf_keras_layer
Γ
Λ	variables
Μtrainable_variables
Νregularization_losses
Ξ	keras_api
Ο__call__
+Π&call_and_return_all_conditional_losses
Ρ_random_generator"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
Ν
Σ	variables
Τtrainable_variables
Υregularization_losses
Φ	keras_api
Χ__call__
+Ψ&call_and_return_all_conditional_losses
	Ωaxis

Ϊgamma
	Ϋbeta"
_tf_keras_layer
ζ
ά	variables
έtrainable_variables
ήregularization_losses
ί	keras_api
ΰ__call__
+α&call_and_return_all_conditional_losses
βkernel
	γbias
!δ_jit_compiled_convolution_op"
_tf_keras_layer
Γ
ε	variables
ζtrainable_variables
ηregularization_losses
θ	keras_api
ι__call__
+κ&call_and_return_all_conditional_losses
λ_random_generator"
_tf_keras_layer
ζ
μ	variables
νtrainable_variables
ξregularization_losses
ο	keras_api
π__call__
+ρ&call_and_return_all_conditional_losses
ςkernel
	σbias
!τ_jit_compiled_convolution_op"
_tf_keras_layer
)
υ	keras_api"
_tf_keras_layer
Ν
φ	variables
χtrainable_variables
ψregularization_losses
ω	keras_api
ϊ__call__
+ϋ&call_and_return_all_conditional_losses
	όaxis

ύgamma
	ώbeta"
_tf_keras_layer
°
?	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_attention_axes
_query_dense

_key_dense
_value_dense
_softmax
_dropout_layer
_output_dense"
_tf_keras_layer
Γ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator"
_tf_keras_layer
)
	keras_api"
_tf_keras_layer
Ν
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis

gamma
	beta"
_tf_keras_layer
ζ
	variables
trainable_variables
regularization_losses
 	keras_api
‘__call__
+’&call_and_return_all_conditional_losses
£kernel
	€bias
!₯_jit_compiled_convolution_op"
_tf_keras_layer
Γ
¦	variables
§trainable_variables
¨regularization_losses
©	keras_api
ͺ__call__
+«&call_and_return_all_conditional_losses
¬_random_generator"
_tf_keras_layer
ζ
­	variables
?trainable_variables
―regularization_losses
°	keras_api
±__call__
+²&call_and_return_all_conditional_losses
³kernel
	΄bias
!΅_jit_compiled_convolution_op"
_tf_keras_layer
)
Ά	keras_api"
_tf_keras_layer
«
·	variables
Έtrainable_variables
Ήregularization_losses
Ί	keras_api
»__call__
+Ό&call_and_return_all_conditional_losses"
_tf_keras_layer
Γ
½	variables
Ύtrainable_variables
Ώregularization_losses
ΐ	keras_api
Α__call__
+Β&call_and_return_all_conditional_losses
Γkernel
	Δbias"
_tf_keras_layer
Γ
Ε	variables
Ζtrainable_variables
Ηregularization_losses
Θ	keras_api
Ι__call__
+Κ&call_and_return_all_conditional_losses
Λ_random_generator"
_tf_keras_layer
Γ
Μ	variables
Νtrainable_variables
Ξregularization_losses
Ο	keras_api
Π__call__
+Ρ&call_and_return_all_conditional_losses
?kernel
	Σbias"
_tf_keras_layer
π
:0
;1
Τ2
Υ3
Φ4
Χ5
Ψ6
Ω7
Ϊ8
Ϋ9
X10
Y11
`12
a13
p14
q15
{16
|17
ά18
έ19
ή20
ί21
ΰ22
α23
β24
γ25
26
27
‘28
’29
±30
²31
Ό32
½33
δ34
ε35
ζ36
η37
θ38
ι39
κ40
λ41
Ϊ42
Ϋ43
β44
γ45
ς46
σ47
ύ48
ώ49
μ50
ν51
ξ52
ο53
π54
ρ55
ς56
σ57
58
59
£60
€61
³62
΄63
Γ64
Δ65
?66
Σ67"
trackable_list_wrapper
π
:0
;1
Τ2
Υ3
Φ4
Χ5
Ψ6
Ω7
Ϊ8
Ϋ9
X10
Y11
`12
a13
p14
q15
{16
|17
ά18
έ19
ή20
ί21
ΰ22
α23
β24
γ25
26
27
‘28
’29
±30
²31
Ό32
½33
δ34
ε35
ζ36
η37
θ38
ι39
κ40
λ41
Ϊ42
Ϋ43
β44
γ45
ς46
σ47
ύ48
ώ49
μ50
ν51
ξ52
ο53
π54
ρ55
ς56
σ57
58
59
£60
€61
³62
΄63
Γ64
Δ65
?66
Σ67"
trackable_list_wrapper
 "
trackable_list_wrapper
Ο
τnon_trainable_variables
υlayers
φmetrics
 χlayer_regularization_losses
ψlayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
0_default_save_signature
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
?
ωtrace_0
ϊtrace_1
ϋtrace_2
όtrace_32ί
%__inference_model_layer_call_fn_14343
%__inference_model_layer_call_fn_16331
%__inference_model_layer_call_fn_16472
%__inference_model_layer_call_fn_15681ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zωtrace_0zϊtrace_1zϋtrace_2zόtrace_3
Ύ
ύtrace_0
ώtrace_1
?trace_2
trace_32Λ
@__inference_model_layer_call_and_return_conditional_losses_16864
@__inference_model_layer_call_and_return_conditional_losses_17347
@__inference_model_layer_call_and_return_conditional_losses_15861
@__inference_model_layer_call_and_return_conditional_losses_16041ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zύtrace_0zώtrace_1z?trace_2ztrace_3
ΛBΘ
 __inference__wrapped_model_13470input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Α
	iter

decay
learning_rate
momentum
rho
:rmsΈ
;rmsΉ
XrmsΊ
Yrms»
`rmsΌ
arms½
prmsΎ
qrmsΏ
{rmsΐ
|rmsΑrmsΒrmsΓ‘rmsΔ’rmsΕ±rmsΖ²rmsΗΌrmsΘ½rmsΙΪrmsΚΫrmsΛβrmsΜγrmsΝςrmsΞσrmsΟύrmsΠώrmsΡrms?rmsΣ£rmsΤ€rmsΥ³rmsΦ΄rmsΧΓrmsΨΔrmsΩ?rmsΪΣrmsΫΤrmsάΥrmsέΦrmsήΧrmsίΨrmsΰΩrmsαΪrmsβΫrmsγάrmsδέrmsεήrmsζίrmsηΰrmsθαrmsιβrmsκγrmsλδrmsμεrmsνζrmsξηrmsοθrmsπιrmsρκrmsςλrmsσμrmsτνrmsυξrmsφοrmsχπrmsψρrmsωςrmsϊσrmsϋ"
	optimizer
-
serving_default"
signature_map
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
ω
trace_02Ϊ
3__inference_layer_normalization_layer_call_fn_17356’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0

trace_02υ
N__inference_layer_normalization_layer_call_and_return_conditional_losses_17378’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0
 "
trackable_list_wrapper
':%2layer_normalization/gamma
&:$2layer_normalization/beta
`
Τ0
Υ1
Φ2
Χ3
Ψ4
Ω5
Ϊ6
Ϋ7"
trackable_list_wrapper
`
Τ0
Υ1
Φ2
Χ3
Ψ4
Ω5
Ϊ6
Ϋ7"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
¦
trace_0
trace_12λ
4__inference_multi_head_attention_layer_call_fn_17400
4__inference_multi_head_attention_layer_call_fn_17422ό
σ²ο
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 ztrace_0ztrace_1
ά
trace_0
trace_12‘
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_17457
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_17499ό
σ²ο
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 ztrace_0ztrace_1
 "
trackable_list_wrapper
φ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
partial_output_shape
full_output_shape
Τkernel
	Υbias"
_tf_keras_layer
φ
	variables
 trainable_variables
‘regularization_losses
’	keras_api
£__call__
+€&call_and_return_all_conditional_losses
₯partial_output_shape
¦full_output_shape
Φkernel
	Χbias"
_tf_keras_layer
φ
§	variables
¨trainable_variables
©regularization_losses
ͺ	keras_api
«__call__
+¬&call_and_return_all_conditional_losses
­partial_output_shape
?full_output_shape
Ψkernel
	Ωbias"
_tf_keras_layer
«
―	variables
°trainable_variables
±regularization_losses
²	keras_api
³__call__
+΄&call_and_return_all_conditional_losses"
_tf_keras_layer
Γ
΅	variables
Άtrainable_variables
·regularization_losses
Έ	keras_api
Ή__call__
+Ί&call_and_return_all_conditional_losses
»_random_generator"
_tf_keras_layer
φ
Ό	variables
½trainable_variables
Ύregularization_losses
Ώ	keras_api
ΐ__call__
+Α&call_and_return_all_conditional_losses
Βpartial_output_shape
Γfull_output_shape
Ϊkernel
	Ϋbias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Δnon_trainable_variables
Εlayers
Ζmetrics
 Ηlayer_regularization_losses
Θlayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
Δ
Ιtrace_0
Κtrace_12
'__inference_dropout_layer_call_fn_17504
'__inference_dropout_layer_call_fn_17509΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zΙtrace_0zΚtrace_1
ϊ
Λtrace_0
Μtrace_12Ώ
B__inference_dropout_layer_call_and_return_conditional_losses_17514
B__inference_dropout_layer_call_and_return_conditional_losses_17526΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zΛtrace_0zΜtrace_1
"
_generic_user_object
"
_generic_user_object
.
X0
Y1"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Νnon_trainable_variables
Ξlayers
Οmetrics
 Πlayer_regularization_losses
Ρlayer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
ϋ
?trace_02ά
5__inference_layer_normalization_1_layer_call_fn_17535’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 z?trace_0

Σtrace_02χ
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_17557’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zΣtrace_0
 "
trackable_list_wrapper
):'2layer_normalization_1/gamma
(:&2layer_normalization_1/beta
.
`0
a1"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Τnon_trainable_variables
Υlayers
Φmetrics
 Χlayer_regularization_losses
Ψlayer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
μ
Ωtrace_02Ν
&__inference_conv1d_layer_call_fn_17566’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zΩtrace_0

Ϊtrace_02θ
A__inference_conv1d_layer_call_and_return_conditional_losses_17582’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zΪtrace_0
#:!2conv1d/kernel
:2conv1d/bias
΄2±?
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ϋnon_trainable_variables
άlayers
έmetrics
 ήlayer_regularization_losses
ίlayer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
Θ
ΰtrace_0
αtrace_12
)__inference_dropout_1_layer_call_fn_17587
)__inference_dropout_1_layer_call_fn_17592΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zΰtrace_0zαtrace_1
ώ
βtrace_0
γtrace_12Γ
D__inference_dropout_1_layer_call_and_return_conditional_losses_17597
D__inference_dropout_1_layer_call_and_return_conditional_losses_17609΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zβtrace_0zγtrace_1
"
_generic_user_object
.
p0
q1"
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
δnon_trainable_variables
εlayers
ζmetrics
 ηlayer_regularization_losses
θlayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
ξ
ιtrace_02Ο
(__inference_conv1d_1_layer_call_fn_17618’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zιtrace_0

κtrace_02κ
C__inference_conv1d_1_layer_call_and_return_conditional_losses_17633’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zκtrace_0
%:#2conv1d_1/kernel
:2conv1d_1/bias
΄2±?
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 0
"
_generic_user_object
.
{0
|1"
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
λnon_trainable_variables
μlayers
νmetrics
 ξlayer_regularization_losses
οlayer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
ϋ
πtrace_02ά
5__inference_layer_normalization_2_layer_call_fn_17642’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zπtrace_0

ρtrace_02χ
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_17664’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zρtrace_0
 "
trackable_list_wrapper
):'2layer_normalization_2/gamma
(:&2layer_normalization_2/beta
`
ά0
έ1
ή2
ί3
ΰ4
α5
β6
γ7"
trackable_list_wrapper
`
ά0
έ1
ή2
ί3
ΰ4
α5
β6
γ7"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
ςnon_trainable_variables
σlayers
τmetrics
 υlayer_regularization_losses
φlayer_metrics
}	variables
~trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ͺ
χtrace_0
ψtrace_12ο
6__inference_multi_head_attention_1_layer_call_fn_17686
6__inference_multi_head_attention_1_layer_call_fn_17708ό
σ²ο
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zχtrace_0zψtrace_1
ΰ
ωtrace_0
ϊtrace_12₯
Q__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_17743
Q__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_17785ό
σ²ο
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zωtrace_0zϊtrace_1
 "
trackable_list_wrapper
φ
ϋ	variables
όtrainable_variables
ύregularization_losses
ώ	keras_api
?__call__
+&call_and_return_all_conditional_losses
partial_output_shape
full_output_shape
άkernel
	έbias"
_tf_keras_layer
φ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
partial_output_shape
full_output_shape
ήkernel
	ίbias"
_tf_keras_layer
φ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
partial_output_shape
full_output_shape
ΰkernel
	αbias"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Γ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator"
_tf_keras_layer
φ
 	variables
‘trainable_variables
’regularization_losses
£	keras_api
€__call__
+₯&call_and_return_all_conditional_losses
¦partial_output_shape
§full_output_shape
βkernel
	γbias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
¨non_trainable_variables
©layers
ͺmetrics
 «layer_regularization_losses
¬layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Θ
­trace_0
?trace_12
)__inference_dropout_2_layer_call_fn_17790
)__inference_dropout_2_layer_call_fn_17795΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 z­trace_0z?trace_1
ώ
―trace_0
°trace_12Γ
D__inference_dropout_2_layer_call_and_return_conditional_losses_17800
D__inference_dropout_2_layer_call_and_return_conditional_losses_17812΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 z―trace_0z°trace_1
"
_generic_user_object
"
_generic_user_object
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
±non_trainable_variables
²layers
³metrics
 ΄layer_regularization_losses
΅layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ϋ
Άtrace_02ά
5__inference_layer_normalization_3_layer_call_fn_17821’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zΆtrace_0

·trace_02χ
P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_17843’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 z·trace_0
 "
trackable_list_wrapper
):'2layer_normalization_3/gamma
(:&2layer_normalization_3/beta
0
‘0
’1"
trackable_list_wrapper
0
‘0
’1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Έnon_trainable_variables
Ήlayers
Ίmetrics
 »layer_regularization_losses
Όlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
ξ
½trace_02Ο
(__inference_conv1d_2_layer_call_fn_17852’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 z½trace_0

Ύtrace_02κ
C__inference_conv1d_2_layer_call_and_return_conditional_losses_17868’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zΎtrace_0
%:#2conv1d_2/kernel
:2conv1d_2/bias
΄2±?
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Ώnon_trainable_variables
ΐlayers
Αmetrics
 Βlayer_regularization_losses
Γlayer_metrics
€	variables
₯trainable_variables
¦regularization_losses
¨__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
Θ
Δtrace_0
Εtrace_12
)__inference_dropout_3_layer_call_fn_17873
)__inference_dropout_3_layer_call_fn_17878΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zΔtrace_0zΕtrace_1
ώ
Ζtrace_0
Ηtrace_12Γ
D__inference_dropout_3_layer_call_and_return_conditional_losses_17883
D__inference_dropout_3_layer_call_and_return_conditional_losses_17895΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zΖtrace_0zΗtrace_1
"
_generic_user_object
0
±0
²1"
trackable_list_wrapper
0
±0
²1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Θnon_trainable_variables
Ιlayers
Κmetrics
 Λlayer_regularization_losses
Μlayer_metrics
«	variables
¬trainable_variables
­regularization_losses
―__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
ξ
Νtrace_02Ο
(__inference_conv1d_3_layer_call_fn_17904’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zΝtrace_0

Ξtrace_02κ
C__inference_conv1d_3_layer_call_and_return_conditional_losses_17919’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zΞtrace_0
%:#2conv1d_3/kernel
:2conv1d_3/bias
΄2±?
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 0
"
_generic_user_object
0
Ό0
½1"
trackable_list_wrapper
0
Ό0
½1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Οnon_trainable_variables
Πlayers
Ρmetrics
 ?layer_regularization_losses
Σlayer_metrics
΅	variables
Άtrainable_variables
·regularization_losses
Ή__call__
+Ί&call_and_return_all_conditional_losses
'Ί"call_and_return_conditional_losses"
_generic_user_object
ϋ
Τtrace_02ά
5__inference_layer_normalization_4_layer_call_fn_17928’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zΤtrace_0

Υtrace_02χ
P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_17950’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zΥtrace_0
 "
trackable_list_wrapper
):'2layer_normalization_4/gamma
(:&2layer_normalization_4/beta
`
δ0
ε1
ζ2
η3
θ4
ι5
κ6
λ7"
trackable_list_wrapper
`
δ0
ε1
ζ2
η3
θ4
ι5
κ6
λ7"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Φnon_trainable_variables
Χlayers
Ψmetrics
 Ωlayer_regularization_losses
Ϊlayer_metrics
Ύ	variables
Ώtrainable_variables
ΐregularization_losses
Β__call__
+Γ&call_and_return_all_conditional_losses
'Γ"call_and_return_conditional_losses"
_generic_user_object
ͺ
Ϋtrace_0
άtrace_12ο
6__inference_multi_head_attention_2_layer_call_fn_17972
6__inference_multi_head_attention_2_layer_call_fn_17994ό
σ²ο
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zΫtrace_0zάtrace_1
ΰ
έtrace_0
ήtrace_12₯
Q__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_18029
Q__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_18071ό
σ²ο
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zέtrace_0zήtrace_1
 "
trackable_list_wrapper
φ
ί	variables
ΰtrainable_variables
αregularization_losses
β	keras_api
γ__call__
+δ&call_and_return_all_conditional_losses
εpartial_output_shape
ζfull_output_shape
δkernel
	εbias"
_tf_keras_layer
φ
η	variables
θtrainable_variables
ιregularization_losses
κ	keras_api
λ__call__
+μ&call_and_return_all_conditional_losses
νpartial_output_shape
ξfull_output_shape
ζkernel
	ηbias"
_tf_keras_layer
φ
ο	variables
πtrainable_variables
ρregularization_losses
ς	keras_api
σ__call__
+τ&call_and_return_all_conditional_losses
υpartial_output_shape
φfull_output_shape
θkernel
	ιbias"
_tf_keras_layer
«
χ	variables
ψtrainable_variables
ωregularization_losses
ϊ	keras_api
ϋ__call__
+ό&call_and_return_all_conditional_losses"
_tf_keras_layer
Γ
ύ	variables
ώtrainable_variables
?regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator"
_tf_keras_layer
φ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
partial_output_shape
full_output_shape
κkernel
	λbias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Λ	variables
Μtrainable_variables
Νregularization_losses
Ο__call__
+Π&call_and_return_all_conditional_losses
'Π"call_and_return_conditional_losses"
_generic_user_object
Θ
trace_0
trace_12
)__inference_dropout_4_layer_call_fn_18076
)__inference_dropout_4_layer_call_fn_18081΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 ztrace_0ztrace_1
ώ
trace_0
trace_12Γ
D__inference_dropout_4_layer_call_and_return_conditional_losses_18086
D__inference_dropout_4_layer_call_and_return_conditional_losses_18098΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 ztrace_0ztrace_1
"
_generic_user_object
"
_generic_user_object
0
Ϊ0
Ϋ1"
trackable_list_wrapper
0
Ϊ0
Ϋ1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Σ	variables
Τtrainable_variables
Υregularization_losses
Χ__call__
+Ψ&call_and_return_all_conditional_losses
'Ψ"call_and_return_conditional_losses"
_generic_user_object
ϋ
trace_02ά
5__inference_layer_normalization_5_layer_call_fn_18107’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0

trace_02χ
P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_18129’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0
 "
trackable_list_wrapper
):'2layer_normalization_5/gamma
(:&2layer_normalization_5/beta
0
β0
γ1"
trackable_list_wrapper
0
β0
γ1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
ά	variables
έtrainable_variables
ήregularization_losses
ΰ__call__
+α&call_and_return_all_conditional_losses
'α"call_and_return_conditional_losses"
_generic_user_object
ξ
‘trace_02Ο
(__inference_conv1d_4_layer_call_fn_18138’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 z‘trace_0

’trace_02κ
C__inference_conv1d_4_layer_call_and_return_conditional_losses_18154’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 z’trace_0
%:#2conv1d_4/kernel
:2conv1d_4/bias
΄2±?
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
£non_trainable_variables
€layers
₯metrics
 ¦layer_regularization_losses
§layer_metrics
ε	variables
ζtrainable_variables
ηregularization_losses
ι__call__
+κ&call_and_return_all_conditional_losses
'κ"call_and_return_conditional_losses"
_generic_user_object
Θ
¨trace_0
©trace_12
)__inference_dropout_5_layer_call_fn_18159
)__inference_dropout_5_layer_call_fn_18164΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 z¨trace_0z©trace_1
ώ
ͺtrace_0
«trace_12Γ
D__inference_dropout_5_layer_call_and_return_conditional_losses_18169
D__inference_dropout_5_layer_call_and_return_conditional_losses_18181΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zͺtrace_0z«trace_1
"
_generic_user_object
0
ς0
σ1"
trackable_list_wrapper
0
ς0
σ1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
¬non_trainable_variables
­layers
?metrics
 ―layer_regularization_losses
°layer_metrics
μ	variables
νtrainable_variables
ξregularization_losses
π__call__
+ρ&call_and_return_all_conditional_losses
'ρ"call_and_return_conditional_losses"
_generic_user_object
ξ
±trace_02Ο
(__inference_conv1d_5_layer_call_fn_18190’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 z±trace_0

²trace_02κ
C__inference_conv1d_5_layer_call_and_return_conditional_losses_18205’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 z²trace_0
%:#2conv1d_5/kernel
:2conv1d_5/bias
΄2±?
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 0
"
_generic_user_object
0
ύ0
ώ1"
trackable_list_wrapper
0
ύ0
ώ1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
³non_trainable_variables
΄layers
΅metrics
 Άlayer_regularization_losses
·layer_metrics
φ	variables
χtrainable_variables
ψregularization_losses
ϊ__call__
+ϋ&call_and_return_all_conditional_losses
'ϋ"call_and_return_conditional_losses"
_generic_user_object
ϋ
Έtrace_02ά
5__inference_layer_normalization_6_layer_call_fn_18214’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zΈtrace_0

Ήtrace_02χ
P__inference_layer_normalization_6_layer_call_and_return_conditional_losses_18236’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zΉtrace_0
 "
trackable_list_wrapper
):'2layer_normalization_6/gamma
(:&2layer_normalization_6/beta
`
μ0
ν1
ξ2
ο3
π4
ρ5
ς6
σ7"
trackable_list_wrapper
`
μ0
ν1
ξ2
ο3
π4
ρ5
ς6
σ7"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Ίnon_trainable_variables
»layers
Όmetrics
 ½layer_regularization_losses
Ύlayer_metrics
?	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ͺ
Ώtrace_0
ΐtrace_12ο
6__inference_multi_head_attention_3_layer_call_fn_18258
6__inference_multi_head_attention_3_layer_call_fn_18280ό
σ²ο
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zΏtrace_0zΐtrace_1
ΰ
Αtrace_0
Βtrace_12₯
Q__inference_multi_head_attention_3_layer_call_and_return_conditional_losses_18315
Q__inference_multi_head_attention_3_layer_call_and_return_conditional_losses_18357ό
σ²ο
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zΑtrace_0zΒtrace_1
 "
trackable_list_wrapper
φ
Γ	variables
Δtrainable_variables
Εregularization_losses
Ζ	keras_api
Η__call__
+Θ&call_and_return_all_conditional_losses
Ιpartial_output_shape
Κfull_output_shape
μkernel
	νbias"
_tf_keras_layer
φ
Λ	variables
Μtrainable_variables
Νregularization_losses
Ξ	keras_api
Ο__call__
+Π&call_and_return_all_conditional_losses
Ρpartial_output_shape
?full_output_shape
ξkernel
	οbias"
_tf_keras_layer
φ
Σ	variables
Τtrainable_variables
Υregularization_losses
Φ	keras_api
Χ__call__
+Ψ&call_and_return_all_conditional_losses
Ωpartial_output_shape
Ϊfull_output_shape
πkernel
	ρbias"
_tf_keras_layer
«
Ϋ	variables
άtrainable_variables
έregularization_losses
ή	keras_api
ί__call__
+ΰ&call_and_return_all_conditional_losses"
_tf_keras_layer
Γ
α	variables
βtrainable_variables
γregularization_losses
δ	keras_api
ε__call__
+ζ&call_and_return_all_conditional_losses
η_random_generator"
_tf_keras_layer
φ
θ	variables
ιtrainable_variables
κregularization_losses
λ	keras_api
μ__call__
+ν&call_and_return_all_conditional_losses
ξpartial_output_shape
οfull_output_shape
ςkernel
	σbias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
πnon_trainable_variables
ρlayers
ςmetrics
 σlayer_regularization_losses
τlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Θ
υtrace_0
φtrace_12
)__inference_dropout_6_layer_call_fn_18362
)__inference_dropout_6_layer_call_fn_18367΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zυtrace_0zφtrace_1
ώ
χtrace_0
ψtrace_12Γ
D__inference_dropout_6_layer_call_and_return_conditional_losses_18372
D__inference_dropout_6_layer_call_and_return_conditional_losses_18384΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zχtrace_0zψtrace_1
"
_generic_user_object
"
_generic_user_object
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
ωnon_trainable_variables
ϊlayers
ϋmetrics
 όlayer_regularization_losses
ύlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ϋ
ώtrace_02ά
5__inference_layer_normalization_7_layer_call_fn_18393’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zώtrace_0

?trace_02χ
P__inference_layer_normalization_7_layer_call_and_return_conditional_losses_18415’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 z?trace_0
 "
trackable_list_wrapper
):'2layer_normalization_7/gamma
(:&2layer_normalization_7/beta
0
£0
€1"
trackable_list_wrapper
0
£0
€1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
‘__call__
+’&call_and_return_all_conditional_losses
'’"call_and_return_conditional_losses"
_generic_user_object
ξ
trace_02Ο
(__inference_conv1d_6_layer_call_fn_18424’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0

trace_02κ
C__inference_conv1d_6_layer_call_and_return_conditional_losses_18440’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0
%:#2conv1d_6/kernel
:2conv1d_6/bias
΄2±?
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¦	variables
§trainable_variables
¨regularization_losses
ͺ__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
Θ
trace_0
trace_12
)__inference_dropout_7_layer_call_fn_18445
)__inference_dropout_7_layer_call_fn_18450΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 ztrace_0ztrace_1
ώ
trace_0
trace_12Γ
D__inference_dropout_7_layer_call_and_return_conditional_losses_18455
D__inference_dropout_7_layer_call_and_return_conditional_losses_18467΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 ztrace_0ztrace_1
"
_generic_user_object
0
³0
΄1"
trackable_list_wrapper
0
³0
΄1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
­	variables
?trainable_variables
―regularization_losses
±__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses"
_generic_user_object
ξ
trace_02Ο
(__inference_conv1d_7_layer_call_fn_18476’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0

trace_02κ
C__inference_conv1d_7_layer_call_and_return_conditional_losses_18491’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0
%:#2conv1d_7/kernel
:2conv1d_7/bias
΄2±?
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 0
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
·	variables
Έtrainable_variables
Ήregularization_losses
»__call__
+Ό&call_and_return_all_conditional_losses
'Ό"call_and_return_conditional_losses"
_generic_user_object

trace_02μ
8__inference_global_average_pooling1d_layer_call_fn_18496―
¦²’
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults’

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0
¦
trace_02
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_18502―
¦²’
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults’

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0
0
Γ0
Δ1"
trackable_list_wrapper
0
Γ0
Δ1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
non_trainable_variables
layers
 metrics
 ‘layer_regularization_losses
’layer_metrics
½	variables
Ύtrainable_variables
Ώregularization_losses
Α__call__
+Β&call_and_return_all_conditional_losses
'Β"call_and_return_conditional_losses"
_generic_user_object
λ
£trace_02Μ
%__inference_dense_layer_call_fn_18511’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 z£trace_0

€trace_02η
@__inference_dense_layer_call_and_return_conditional_losses_18522’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 z€trace_0
 :
Θ2dense/kernel
:2
dense/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
₯non_trainable_variables
¦layers
§metrics
 ¨layer_regularization_losses
©layer_metrics
Ε	variables
Ζtrainable_variables
Ηregularization_losses
Ι__call__
+Κ&call_and_return_all_conditional_losses
'Κ"call_and_return_conditional_losses"
_generic_user_object
Θ
ͺtrace_0
«trace_12
)__inference_dropout_8_layer_call_fn_18527
)__inference_dropout_8_layer_call_fn_18532΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zͺtrace_0z«trace_1
ώ
¬trace_0
­trace_12Γ
D__inference_dropout_8_layer_call_and_return_conditional_losses_18537
D__inference_dropout_8_layer_call_and_return_conditional_losses_18549΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 z¬trace_0z­trace_1
"
_generic_user_object
0
?0
Σ1"
trackable_list_wrapper
0
?0
Σ1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
?non_trainable_variables
―layers
°metrics
 ±layer_regularization_losses
²layer_metrics
Μ	variables
Νtrainable_variables
Ξregularization_losses
Π__call__
+Ρ&call_and_return_all_conditional_losses
'Ρ"call_and_return_conditional_losses"
_generic_user_object
ν
³trace_02Ξ
'__inference_dense_1_layer_call_fn_18558’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 z³trace_0

΄trace_02ι
B__inference_dense_1_layer_call_and_return_conditional_losses_18568’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 z΄trace_0
!:	2dense_1/kernel
:2dense_1/bias
7:522!multi_head_attention/query/kernel
1:/22multi_head_attention/query/bias
5:322multi_head_attention/key/kernel
/:-22multi_head_attention/key/bias
7:522!multi_head_attention/value/kernel
1:/22multi_head_attention/value/bias
B:@22,multi_head_attention/attention_output/kernel
8:62*multi_head_attention/attention_output/bias
9:722#multi_head_attention_1/query/kernel
3:122!multi_head_attention_1/query/bias
7:522!multi_head_attention_1/key/kernel
1:/22multi_head_attention_1/key/bias
9:722#multi_head_attention_1/value/kernel
3:122!multi_head_attention_1/value/bias
D:B22.multi_head_attention_1/attention_output/kernel
::82,multi_head_attention_1/attention_output/bias
9:722#multi_head_attention_2/query/kernel
3:122!multi_head_attention_2/query/bias
7:522!multi_head_attention_2/key/kernel
1:/22multi_head_attention_2/key/bias
9:722#multi_head_attention_2/value/kernel
3:122!multi_head_attention_2/value/bias
D:B22.multi_head_attention_2/attention_output/kernel
::82,multi_head_attention_2/attention_output/bias
9:722#multi_head_attention_3/query/kernel
3:122!multi_head_attention_3/query/bias
7:522!multi_head_attention_3/key/kernel
1:/22multi_head_attention_3/key/bias
9:722#multi_head_attention_3/value/kernel
3:122!multi_head_attention_3/value/bias
D:B22.multi_head_attention_3/attention_output/kernel
::82,multi_head_attention_3/attention_output/bias
 "
trackable_list_wrapper
ή
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40"
trackable_list_wrapper
0
΅0
Ά1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ψBυ
%__inference_model_layer_call_fn_14343input_1"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
χBτ
%__inference_model_layer_call_fn_16331inputs"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
χBτ
%__inference_model_layer_call_fn_16472inputs"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ψBυ
%__inference_model_layer_call_fn_15681input_1"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
@__inference_model_layer_call_and_return_conditional_losses_16864inputs"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
@__inference_model_layer_call_and_return_conditional_losses_17347inputs"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
@__inference_model_layer_call_and_return_conditional_losses_15861input_1"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
@__inference_model_layer_call_and_return_conditional_losses_16041input_1"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
:	 (2iter
: (2decay
: (2learning_rate
: (2momentum
: (2rho
ΚBΗ
#__inference_signature_wrapper_16190input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
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
ηBδ
3__inference_layer_normalization_layer_call_fn_17356inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
B?
N__inference_layer_normalization_layer_call_and_return_conditional_losses_17378inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
J
C0
D1
E2
F3
G4
H5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ΘBΕ
4__inference_multi_head_attention_layer_call_fn_17400queryvalue"ό
σ²ο
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ΘBΕ
4__inference_multi_head_attention_layer_call_fn_17422queryvalue"ό
σ²ο
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
γBΰ
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_17457queryvalue"ό
σ²ο
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
γBΰ
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_17499queryvalue"ό
σ²ο
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
0
Τ0
Υ1"
trackable_list_wrapper
0
Τ0
Υ1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
·non_trainable_variables
Έlayers
Ήmetrics
 Ίlayer_regularization_losses
»layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Φ0
Χ1"
trackable_list_wrapper
0
Φ0
Χ1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Όnon_trainable_variables
½layers
Ύmetrics
 Ώlayer_regularization_losses
ΐlayer_metrics
	variables
 trainable_variables
‘regularization_losses
£__call__
+€&call_and_return_all_conditional_losses
'€"call_and_return_conditional_losses"
_generic_user_object
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ψ0
Ω1"
trackable_list_wrapper
0
Ψ0
Ω1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Αnon_trainable_variables
Βlayers
Γmetrics
 Δlayer_regularization_losses
Εlayer_metrics
§	variables
¨trainable_variables
©regularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Ζnon_trainable_variables
Ηlayers
Θmetrics
 Ιlayer_regularization_losses
Κlayer_metrics
―	variables
°trainable_variables
±regularization_losses
³__call__
+΄&call_and_return_all_conditional_losses
'΄"call_and_return_conditional_losses"
_generic_user_object
΅2²―
¦²’
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults’

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
΅2²―
¦²’
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults’

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Λnon_trainable_variables
Μlayers
Νmetrics
 Ξlayer_regularization_losses
Οlayer_metrics
΅	variables
Άtrainable_variables
·regularization_losses
Ή__call__
+Ί&call_and_return_all_conditional_losses
'Ί"call_and_return_conditional_losses"
_generic_user_object
Ί2·΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Ί2·΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
"
_generic_user_object
0
Ϊ0
Ϋ1"
trackable_list_wrapper
0
Ϊ0
Ϋ1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Πnon_trainable_variables
Ρlayers
?metrics
 Σlayer_regularization_losses
Τlayer_metrics
Ό	variables
½trainable_variables
Ύregularization_losses
ΐ__call__
+Α&call_and_return_all_conditional_losses
'Α"call_and_return_conditional_losses"
_generic_user_object
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
νBκ
'__inference_dropout_layer_call_fn_17504inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
νBκ
'__inference_dropout_layer_call_fn_17509inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
B__inference_dropout_layer_call_and_return_conditional_losses_17514inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
B__inference_dropout_layer_call_and_return_conditional_losses_17526inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
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
ιBζ
5__inference_layer_normalization_1_layer_call_fn_17535inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
B
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_17557inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
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
ΪBΧ
&__inference_conv1d_layer_call_fn_17566inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
υBς
A__inference_conv1d_layer_call_and_return_conditional_losses_17582inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
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
οBμ
)__inference_dropout_1_layer_call_fn_17587inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
οBμ
)__inference_dropout_1_layer_call_fn_17592inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
D__inference_dropout_1_layer_call_and_return_conditional_losses_17597inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
D__inference_dropout_1_layer_call_and_return_conditional_losses_17609inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
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
άBΩ
(__inference_conv1d_1_layer_call_fn_17618inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
χBτ
C__inference_conv1d_1_layer_call_and_return_conditional_losses_17633inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
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
ιBζ
5__inference_layer_normalization_2_layer_call_fn_17642inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
B
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_17664inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
P
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ΚBΗ
6__inference_multi_head_attention_1_layer_call_fn_17686queryvalue"ό
σ²ο
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ΚBΗ
6__inference_multi_head_attention_1_layer_call_fn_17708queryvalue"ό
σ²ο
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
εBβ
Q__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_17743queryvalue"ό
σ²ο
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
εBβ
Q__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_17785queryvalue"ό
σ²ο
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
0
ά0
έ1"
trackable_list_wrapper
0
ά0
έ1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Υnon_trainable_variables
Φlayers
Χmetrics
 Ψlayer_regularization_losses
Ωlayer_metrics
ϋ	variables
όtrainable_variables
ύregularization_losses
?__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
ή0
ί1"
trackable_list_wrapper
0
ή0
ί1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Ϊnon_trainable_variables
Ϋlayers
άmetrics
 έlayer_regularization_losses
ήlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
ΰ0
α1"
trackable_list_wrapper
0
ΰ0
α1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
ίnon_trainable_variables
ΰlayers
αmetrics
 βlayer_regularization_losses
γlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
δnon_trainable_variables
εlayers
ζmetrics
 ηlayer_regularization_losses
θlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
΅2²―
¦²’
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults’

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
΅2²―
¦²’
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults’

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
ιnon_trainable_variables
κlayers
λmetrics
 μlayer_regularization_losses
νlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ί2·΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Ί2·΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
"
_generic_user_object
0
β0
γ1"
trackable_list_wrapper
0
β0
γ1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
ξnon_trainable_variables
οlayers
πmetrics
 ρlayer_regularization_losses
ςlayer_metrics
 	variables
‘trainable_variables
’regularization_losses
€__call__
+₯&call_and_return_all_conditional_losses
'₯"call_and_return_conditional_losses"
_generic_user_object
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
οBμ
)__inference_dropout_2_layer_call_fn_17790inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
οBμ
)__inference_dropout_2_layer_call_fn_17795inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
D__inference_dropout_2_layer_call_and_return_conditional_losses_17800inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
D__inference_dropout_2_layer_call_and_return_conditional_losses_17812inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
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
ιBζ
5__inference_layer_normalization_3_layer_call_fn_17821inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
B
P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_17843inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
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
άBΩ
(__inference_conv1d_2_layer_call_fn_17852inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
χBτ
C__inference_conv1d_2_layer_call_and_return_conditional_losses_17868inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
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
οBμ
)__inference_dropout_3_layer_call_fn_17873inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
οBμ
)__inference_dropout_3_layer_call_fn_17878inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
D__inference_dropout_3_layer_call_and_return_conditional_losses_17883inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
D__inference_dropout_3_layer_call_and_return_conditional_losses_17895inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
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
άBΩ
(__inference_conv1d_3_layer_call_fn_17904inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
χBτ
C__inference_conv1d_3_layer_call_and_return_conditional_losses_17919inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
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
ιBζ
5__inference_layer_normalization_4_layer_call_fn_17928inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
B
P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_17950inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
P
Ε0
Ζ1
Η2
Θ3
Ι4
Κ5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ΚBΗ
6__inference_multi_head_attention_2_layer_call_fn_17972queryvalue"ό
σ²ο
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ΚBΗ
6__inference_multi_head_attention_2_layer_call_fn_17994queryvalue"ό
σ²ο
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
εBβ
Q__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_18029queryvalue"ό
σ²ο
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
εBβ
Q__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_18071queryvalue"ό
σ²ο
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
0
δ0
ε1"
trackable_list_wrapper
0
δ0
ε1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
σnon_trainable_variables
τlayers
υmetrics
 φlayer_regularization_losses
χlayer_metrics
ί	variables
ΰtrainable_variables
αregularization_losses
γ__call__
+δ&call_and_return_all_conditional_losses
'δ"call_and_return_conditional_losses"
_generic_user_object
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
ζ0
η1"
trackable_list_wrapper
0
ζ0
η1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
ψnon_trainable_variables
ωlayers
ϊmetrics
 ϋlayer_regularization_losses
όlayer_metrics
η	variables
θtrainable_variables
ιregularization_losses
λ__call__
+μ&call_and_return_all_conditional_losses
'μ"call_and_return_conditional_losses"
_generic_user_object
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
θ0
ι1"
trackable_list_wrapper
0
θ0
ι1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
ύnon_trainable_variables
ώlayers
?metrics
 layer_regularization_losses
layer_metrics
ο	variables
πtrainable_variables
ρregularization_losses
σ__call__
+τ&call_and_return_all_conditional_losses
'τ"call_and_return_conditional_losses"
_generic_user_object
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
χ	variables
ψtrainable_variables
ωregularization_losses
ϋ__call__
+ό&call_and_return_all_conditional_losses
'ό"call_and_return_conditional_losses"
_generic_user_object
΅2²―
¦²’
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults’

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
΅2²―
¦²’
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults’

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ύ	variables
ώtrainable_variables
?regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ί2·΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Ί2·΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
"
_generic_user_object
0
κ0
λ1"
trackable_list_wrapper
0
κ0
λ1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
οBμ
)__inference_dropout_4_layer_call_fn_18076inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
οBμ
)__inference_dropout_4_layer_call_fn_18081inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
D__inference_dropout_4_layer_call_and_return_conditional_losses_18086inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
D__inference_dropout_4_layer_call_and_return_conditional_losses_18098inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
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
ιBζ
5__inference_layer_normalization_5_layer_call_fn_18107inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
B
P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_18129inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
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
άBΩ
(__inference_conv1d_4_layer_call_fn_18138inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
χBτ
C__inference_conv1d_4_layer_call_and_return_conditional_losses_18154inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
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
οBμ
)__inference_dropout_5_layer_call_fn_18159inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
οBμ
)__inference_dropout_5_layer_call_fn_18164inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
D__inference_dropout_5_layer_call_and_return_conditional_losses_18169inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
D__inference_dropout_5_layer_call_and_return_conditional_losses_18181inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
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
άBΩ
(__inference_conv1d_5_layer_call_fn_18190inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
χBτ
C__inference_conv1d_5_layer_call_and_return_conditional_losses_18205inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
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
ιBζ
5__inference_layer_normalization_6_layer_call_fn_18214inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
B
P__inference_layer_normalization_6_layer_call_and_return_conditional_losses_18236inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
P
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ΚBΗ
6__inference_multi_head_attention_3_layer_call_fn_18258queryvalue"ό
σ²ο
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ΚBΗ
6__inference_multi_head_attention_3_layer_call_fn_18280queryvalue"ό
σ²ο
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
εBβ
Q__inference_multi_head_attention_3_layer_call_and_return_conditional_losses_18315queryvalue"ό
σ²ο
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
εBβ
Q__inference_multi_head_attention_3_layer_call_and_return_conditional_losses_18357queryvalue"ό
σ²ο
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
0
μ0
ν1"
trackable_list_wrapper
0
μ0
ν1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Γ	variables
Δtrainable_variables
Εregularization_losses
Η__call__
+Θ&call_and_return_all_conditional_losses
'Θ"call_and_return_conditional_losses"
_generic_user_object
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
ξ0
ο1"
trackable_list_wrapper
0
ξ0
ο1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Λ	variables
Μtrainable_variables
Νregularization_losses
Ο__call__
+Π&call_and_return_all_conditional_losses
'Π"call_and_return_conditional_losses"
_generic_user_object
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
π0
ρ1"
trackable_list_wrapper
0
π0
ρ1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Σ	variables
Τtrainable_variables
Υregularization_losses
Χ__call__
+Ψ&call_and_return_all_conditional_losses
'Ψ"call_and_return_conditional_losses"
_generic_user_object
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
 non_trainable_variables
‘layers
’metrics
 £layer_regularization_losses
€layer_metrics
Ϋ	variables
άtrainable_variables
έregularization_losses
ί__call__
+ΰ&call_and_return_all_conditional_losses
'ΰ"call_and_return_conditional_losses"
_generic_user_object
΅2²―
¦²’
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults’

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
΅2²―
¦²’
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults’

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
₯non_trainable_variables
¦layers
§metrics
 ¨layer_regularization_losses
©layer_metrics
α	variables
βtrainable_variables
γregularization_losses
ε__call__
+ζ&call_and_return_all_conditional_losses
'ζ"call_and_return_conditional_losses"
_generic_user_object
Ί2·΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Ί2·΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
"
_generic_user_object
0
ς0
σ1"
trackable_list_wrapper
0
ς0
σ1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
ͺnon_trainable_variables
«layers
¬metrics
 ­layer_regularization_losses
?layer_metrics
θ	variables
ιtrainable_variables
κregularization_losses
μ__call__
+ν&call_and_return_all_conditional_losses
'ν"call_and_return_conditional_losses"
_generic_user_object
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
οBμ
)__inference_dropout_6_layer_call_fn_18362inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
οBμ
)__inference_dropout_6_layer_call_fn_18367inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
D__inference_dropout_6_layer_call_and_return_conditional_losses_18372inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
D__inference_dropout_6_layer_call_and_return_conditional_losses_18384inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
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
ιBζ
5__inference_layer_normalization_7_layer_call_fn_18393inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
B
P__inference_layer_normalization_7_layer_call_and_return_conditional_losses_18415inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
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
άBΩ
(__inference_conv1d_6_layer_call_fn_18424inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
χBτ
C__inference_conv1d_6_layer_call_and_return_conditional_losses_18440inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
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
οBμ
)__inference_dropout_7_layer_call_fn_18445inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
οBμ
)__inference_dropout_7_layer_call_fn_18450inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
D__inference_dropout_7_layer_call_and_return_conditional_losses_18455inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
D__inference_dropout_7_layer_call_and_return_conditional_losses_18467inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
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
άBΩ
(__inference_conv1d_7_layer_call_fn_18476inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
χBτ
C__inference_conv1d_7_layer_call_and_return_conditional_losses_18491inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
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
ωBφ
8__inference_global_average_pooling1d_layer_call_fn_18496inputs"―
¦²’
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults’

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
B
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_18502inputs"―
¦²’
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults’

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
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
ΩBΦ
%__inference_dense_layer_call_fn_18511inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
τBρ
@__inference_dense_layer_call_and_return_conditional_losses_18522inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
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
οBμ
)__inference_dropout_8_layer_call_fn_18527inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
οBμ
)__inference_dropout_8_layer_call_fn_18532inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
D__inference_dropout_8_layer_call_and_return_conditional_losses_18537inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
D__inference_dropout_8_layer_call_and_return_conditional_losses_18549inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
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
ΫBΨ
'__inference_dense_1_layer_call_fn_18558inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
φBσ
B__inference_dense_1_layer_call_and_return_conditional_losses_18568inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
R
―	variables
°	keras_api

±total

²count"
_tf_keras_metric
c
³	variables
΄	keras_api

΅total

Άcount
·
_fn_kwargs"
_tf_keras_metric
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
±0
²1"
trackable_list_wrapper
.
―	variables"
_generic_user_object
:  (2total
:  (2count
0
΅0
Ά1"
trackable_list_wrapper
.
³	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
):'2layer_normalization/gamma/rms
(:&2layer_normalization/beta/rms
+:)2layer_normalization_1/gamma/rms
*:(2layer_normalization_1/beta/rms
%:#2conv1d/kernel/rms
:2conv1d/bias/rms
':%2conv1d_1/kernel/rms
:2conv1d_1/bias/rms
+:)2layer_normalization_2/gamma/rms
*:(2layer_normalization_2/beta/rms
+:)2layer_normalization_3/gamma/rms
*:(2layer_normalization_3/beta/rms
':%2conv1d_2/kernel/rms
:2conv1d_2/bias/rms
':%2conv1d_3/kernel/rms
:2conv1d_3/bias/rms
+:)2layer_normalization_4/gamma/rms
*:(2layer_normalization_4/beta/rms
+:)2layer_normalization_5/gamma/rms
*:(2layer_normalization_5/beta/rms
':%2conv1d_4/kernel/rms
:2conv1d_4/bias/rms
':%2conv1d_5/kernel/rms
:2conv1d_5/bias/rms
+:)2layer_normalization_6/gamma/rms
*:(2layer_normalization_6/beta/rms
+:)2layer_normalization_7/gamma/rms
*:(2layer_normalization_7/beta/rms
':%2conv1d_6/kernel/rms
:2conv1d_6/bias/rms
':%2conv1d_7/kernel/rms
:2conv1d_7/bias/rms
": 
Θ2dense/kernel/rms
:2dense/bias/rms
#:!	2dense_1/kernel/rms
:2dense_1/bias/rms
9:722%multi_head_attention/query/kernel/rms
3:122#multi_head_attention/query/bias/rms
7:522#multi_head_attention/key/kernel/rms
1:/22!multi_head_attention/key/bias/rms
9:722%multi_head_attention/value/kernel/rms
3:122#multi_head_attention/value/bias/rms
D:B220multi_head_attention/attention_output/kernel/rms
::82.multi_head_attention/attention_output/bias/rms
;:922'multi_head_attention_1/query/kernel/rms
5:322%multi_head_attention_1/query/bias/rms
9:722%multi_head_attention_1/key/kernel/rms
3:122#multi_head_attention_1/key/bias/rms
;:922'multi_head_attention_1/value/kernel/rms
5:322%multi_head_attention_1/value/bias/rms
F:D222multi_head_attention_1/attention_output/kernel/rms
<::20multi_head_attention_1/attention_output/bias/rms
;:922'multi_head_attention_2/query/kernel/rms
5:322%multi_head_attention_2/query/bias/rms
9:722%multi_head_attention_2/key/kernel/rms
3:122#multi_head_attention_2/key/bias/rms
;:922'multi_head_attention_2/value/kernel/rms
5:322%multi_head_attention_2/value/bias/rms
F:D222multi_head_attention_2/attention_output/kernel/rms
<::20multi_head_attention_2/attention_output/bias/rms
;:922'multi_head_attention_3/query/kernel/rms
5:322%multi_head_attention_3/query/bias/rms
9:722%multi_head_attention_3/key/kernel/rms
3:122#multi_head_attention_3/key/bias/rms
;:922'multi_head_attention_3/value/kernel/rms
5:322%multi_head_attention_3/value/bias/rms
F:D222multi_head_attention_3/attention_output/kernel/rms
<::20multi_head_attention_3/attention_output/bias/rms
 __inference__wrapped_model_13470κ~:;ΤΥΦΧΨΩΪΫXY`apq{|άέήίΰαβγ‘’±²Ό½δεζηθικλΪΫβγςσύώμνξοπρςσ£€³΄ΓΔ?Σ5’2
+’(
&#
input_1?????????Θ
ͺ "1ͺ.
,
dense_1!
dense_1?????????­
C__inference_conv1d_1_layer_call_and_return_conditional_losses_17633fpq4’1
*’'
%"
inputs?????????Θ
ͺ "*’'
 
0?????????Θ
 
(__inference_conv1d_1_layer_call_fn_17618Ypq4’1
*’'
%"
inputs?????????Θ
ͺ "?????????Θ―
C__inference_conv1d_2_layer_call_and_return_conditional_losses_17868h‘’4’1
*’'
%"
inputs?????????Θ
ͺ "*’'
 
0?????????Θ
 
(__inference_conv1d_2_layer_call_fn_17852[‘’4’1
*’'
%"
inputs?????????Θ
ͺ "?????????Θ―
C__inference_conv1d_3_layer_call_and_return_conditional_losses_17919h±²4’1
*’'
%"
inputs?????????Θ
ͺ "*’'
 
0?????????Θ
 
(__inference_conv1d_3_layer_call_fn_17904[±²4’1
*’'
%"
inputs?????????Θ
ͺ "?????????Θ―
C__inference_conv1d_4_layer_call_and_return_conditional_losses_18154hβγ4’1
*’'
%"
inputs?????????Θ
ͺ "*’'
 
0?????????Θ
 
(__inference_conv1d_4_layer_call_fn_18138[βγ4’1
*’'
%"
inputs?????????Θ
ͺ "?????????Θ―
C__inference_conv1d_5_layer_call_and_return_conditional_losses_18205hςσ4’1
*’'
%"
inputs?????????Θ
ͺ "*’'
 
0?????????Θ
 
(__inference_conv1d_5_layer_call_fn_18190[ςσ4’1
*’'
%"
inputs?????????Θ
ͺ "?????????Θ―
C__inference_conv1d_6_layer_call_and_return_conditional_losses_18440h£€4’1
*’'
%"
inputs?????????Θ
ͺ "*’'
 
0?????????Θ
 
(__inference_conv1d_6_layer_call_fn_18424[£€4’1
*’'
%"
inputs?????????Θ
ͺ "?????????Θ―
C__inference_conv1d_7_layer_call_and_return_conditional_losses_18491h³΄4’1
*’'
%"
inputs?????????Θ
ͺ "*’'
 
0?????????Θ
 
(__inference_conv1d_7_layer_call_fn_18476[³΄4’1
*’'
%"
inputs?????????Θ
ͺ "?????????Θ«
A__inference_conv1d_layer_call_and_return_conditional_losses_17582f`a4’1
*’'
%"
inputs?????????Θ
ͺ "*’'
 
0?????????Θ
 
&__inference_conv1d_layer_call_fn_17566Y`a4’1
*’'
%"
inputs?????????Θ
ͺ "?????????Θ₯
B__inference_dense_1_layer_call_and_return_conditional_losses_18568_?Σ0’-
&’#
!
inputs?????????
ͺ "%’"

0?????????
 }
'__inference_dense_1_layer_call_fn_18558R?Σ0’-
&’#
!
inputs?????????
ͺ "?????????€
@__inference_dense_layer_call_and_return_conditional_losses_18522`ΓΔ0’-
&’#
!
inputs?????????Θ
ͺ "&’#

0?????????
 |
%__inference_dense_layer_call_fn_18511SΓΔ0’-
&’#
!
inputs?????????Θ
ͺ "??????????
D__inference_dropout_1_layer_call_and_return_conditional_losses_17597f8’5
.’+
%"
inputs?????????Θ
p 
ͺ "*’'
 
0?????????Θ
 ?
D__inference_dropout_1_layer_call_and_return_conditional_losses_17609f8’5
.’+
%"
inputs?????????Θ
p
ͺ "*’'
 
0?????????Θ
 
)__inference_dropout_1_layer_call_fn_17587Y8’5
.’+
%"
inputs?????????Θ
p 
ͺ "?????????Θ
)__inference_dropout_1_layer_call_fn_17592Y8’5
.’+
%"
inputs?????????Θ
p
ͺ "?????????Θ?
D__inference_dropout_2_layer_call_and_return_conditional_losses_17800f8’5
.’+
%"
inputs?????????Θ
p 
ͺ "*’'
 
0?????????Θ
 ?
D__inference_dropout_2_layer_call_and_return_conditional_losses_17812f8’5
.’+
%"
inputs?????????Θ
p
ͺ "*’'
 
0?????????Θ
 
)__inference_dropout_2_layer_call_fn_17790Y8’5
.’+
%"
inputs?????????Θ
p 
ͺ "?????????Θ
)__inference_dropout_2_layer_call_fn_17795Y8’5
.’+
%"
inputs?????????Θ
p
ͺ "?????????Θ?
D__inference_dropout_3_layer_call_and_return_conditional_losses_17883f8’5
.’+
%"
inputs?????????Θ
p 
ͺ "*’'
 
0?????????Θ
 ?
D__inference_dropout_3_layer_call_and_return_conditional_losses_17895f8’5
.’+
%"
inputs?????????Θ
p
ͺ "*’'
 
0?????????Θ
 
)__inference_dropout_3_layer_call_fn_17873Y8’5
.’+
%"
inputs?????????Θ
p 
ͺ "?????????Θ
)__inference_dropout_3_layer_call_fn_17878Y8’5
.’+
%"
inputs?????????Θ
p
ͺ "?????????Θ?
D__inference_dropout_4_layer_call_and_return_conditional_losses_18086f8’5
.’+
%"
inputs?????????Θ
p 
ͺ "*’'
 
0?????????Θ
 ?
D__inference_dropout_4_layer_call_and_return_conditional_losses_18098f8’5
.’+
%"
inputs?????????Θ
p
ͺ "*’'
 
0?????????Θ
 
)__inference_dropout_4_layer_call_fn_18076Y8’5
.’+
%"
inputs?????????Θ
p 
ͺ "?????????Θ
)__inference_dropout_4_layer_call_fn_18081Y8’5
.’+
%"
inputs?????????Θ
p
ͺ "?????????Θ?
D__inference_dropout_5_layer_call_and_return_conditional_losses_18169f8’5
.’+
%"
inputs?????????Θ
p 
ͺ "*’'
 
0?????????Θ
 ?
D__inference_dropout_5_layer_call_and_return_conditional_losses_18181f8’5
.’+
%"
inputs?????????Θ
p
ͺ "*’'
 
0?????????Θ
 
)__inference_dropout_5_layer_call_fn_18159Y8’5
.’+
%"
inputs?????????Θ
p 
ͺ "?????????Θ
)__inference_dropout_5_layer_call_fn_18164Y8’5
.’+
%"
inputs?????????Θ
p
ͺ "?????????Θ?
D__inference_dropout_6_layer_call_and_return_conditional_losses_18372f8’5
.’+
%"
inputs?????????Θ
p 
ͺ "*’'
 
0?????????Θ
 ?
D__inference_dropout_6_layer_call_and_return_conditional_losses_18384f8’5
.’+
%"
inputs?????????Θ
p
ͺ "*’'
 
0?????????Θ
 
)__inference_dropout_6_layer_call_fn_18362Y8’5
.’+
%"
inputs?????????Θ
p 
ͺ "?????????Θ
)__inference_dropout_6_layer_call_fn_18367Y8’5
.’+
%"
inputs?????????Θ
p
ͺ "?????????Θ?
D__inference_dropout_7_layer_call_and_return_conditional_losses_18455f8’5
.’+
%"
inputs?????????Θ
p 
ͺ "*’'
 
0?????????Θ
 ?
D__inference_dropout_7_layer_call_and_return_conditional_losses_18467f8’5
.’+
%"
inputs?????????Θ
p
ͺ "*’'
 
0?????????Θ
 
)__inference_dropout_7_layer_call_fn_18445Y8’5
.’+
%"
inputs?????????Θ
p 
ͺ "?????????Θ
)__inference_dropout_7_layer_call_fn_18450Y8’5
.’+
%"
inputs?????????Θ
p
ͺ "?????????Θ¦
D__inference_dropout_8_layer_call_and_return_conditional_losses_18537^4’1
*’'
!
inputs?????????
p 
ͺ "&’#

0?????????
 ¦
D__inference_dropout_8_layer_call_and_return_conditional_losses_18549^4’1
*’'
!
inputs?????????
p
ͺ "&’#

0?????????
 ~
)__inference_dropout_8_layer_call_fn_18527Q4’1
*’'
!
inputs?????????
p 
ͺ "?????????~
)__inference_dropout_8_layer_call_fn_18532Q4’1
*’'
!
inputs?????????
p
ͺ "?????????¬
B__inference_dropout_layer_call_and_return_conditional_losses_17514f8’5
.’+
%"
inputs?????????Θ
p 
ͺ "*’'
 
0?????????Θ
 ¬
B__inference_dropout_layer_call_and_return_conditional_losses_17526f8’5
.’+
%"
inputs?????????Θ
p
ͺ "*’'
 
0?????????Θ
 
'__inference_dropout_layer_call_fn_17504Y8’5
.’+
%"
inputs?????????Θ
p 
ͺ "?????????Θ
'__inference_dropout_layer_call_fn_17509Y8’5
.’+
%"
inputs?????????Θ
p
ͺ "?????????Θ?
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_18502{I’F
?’<
63
inputs'???????????????????????????

 
ͺ ".’+
$!
0??????????????????
 ͺ
8__inference_global_average_pooling1d_layer_call_fn_18496nI’F
?’<
63
inputs'???????????????????????????

 
ͺ "!??????????????????Ί
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_17557fXY4’1
*’'
%"
inputs?????????Θ
ͺ "*’'
 
0?????????Θ
 
5__inference_layer_normalization_1_layer_call_fn_17535YXY4’1
*’'
%"
inputs?????????Θ
ͺ "?????????ΘΊ
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_17664f{|4’1
*’'
%"
inputs?????????Θ
ͺ "*’'
 
0?????????Θ
 
5__inference_layer_normalization_2_layer_call_fn_17642Y{|4’1
*’'
%"
inputs?????????Θ
ͺ "?????????ΘΌ
P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_17843h4’1
*’'
%"
inputs?????????Θ
ͺ "*’'
 
0?????????Θ
 
5__inference_layer_normalization_3_layer_call_fn_17821[4’1
*’'
%"
inputs?????????Θ
ͺ "?????????ΘΌ
P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_17950hΌ½4’1
*’'
%"
inputs?????????Θ
ͺ "*’'
 
0?????????Θ
 
5__inference_layer_normalization_4_layer_call_fn_17928[Ό½4’1
*’'
%"
inputs?????????Θ
ͺ "?????????ΘΌ
P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_18129hΪΫ4’1
*’'
%"
inputs?????????Θ
ͺ "*’'
 
0?????????Θ
 
5__inference_layer_normalization_5_layer_call_fn_18107[ΪΫ4’1
*’'
%"
inputs?????????Θ
ͺ "?????????ΘΌ
P__inference_layer_normalization_6_layer_call_and_return_conditional_losses_18236hύώ4’1
*’'
%"
inputs?????????Θ
ͺ "*’'
 
0?????????Θ
 
5__inference_layer_normalization_6_layer_call_fn_18214[ύώ4’1
*’'
%"
inputs?????????Θ
ͺ "?????????ΘΌ
P__inference_layer_normalization_7_layer_call_and_return_conditional_losses_18415h4’1
*’'
%"
inputs?????????Θ
ͺ "*’'
 
0?????????Θ
 
5__inference_layer_normalization_7_layer_call_fn_18393[4’1
*’'
%"
inputs?????????Θ
ͺ "?????????ΘΈ
N__inference_layer_normalization_layer_call_and_return_conditional_losses_17378f:;4’1
*’'
%"
inputs?????????Θ
ͺ "*’'
 
0?????????Θ
 
3__inference_layer_normalization_layer_call_fn_17356Y:;4’1
*’'
%"
inputs?????????Θ
ͺ "?????????Θ«
@__inference_model_layer_call_and_return_conditional_losses_15861ζ~:;ΤΥΦΧΨΩΪΫXY`apq{|άέήίΰαβγ‘’±²Ό½δεζηθικλΪΫβγςσύώμνξοπρςσ£€³΄ΓΔ?Σ=’:
3’0
&#
input_1?????????Θ
p 

 
ͺ "%’"

0?????????
 «
@__inference_model_layer_call_and_return_conditional_losses_16041ζ~:;ΤΥΦΧΨΩΪΫXY`apq{|άέήίΰαβγ‘’±²Ό½δεζηθικλΪΫβγςσύώμνξοπρςσ£€³΄ΓΔ?Σ=’:
3’0
&#
input_1?????????Θ
p

 
ͺ "%’"

0?????????
 ͺ
@__inference_model_layer_call_and_return_conditional_losses_16864ε~:;ΤΥΦΧΨΩΪΫXY`apq{|άέήίΰαβγ‘’±²Ό½δεζηθικλΪΫβγςσύώμνξοπρςσ£€³΄ΓΔ?Σ<’9
2’/
%"
inputs?????????Θ
p 

 
ͺ "%’"

0?????????
 ͺ
@__inference_model_layer_call_and_return_conditional_losses_17347ε~:;ΤΥΦΧΨΩΪΫXY`apq{|άέήίΰαβγ‘’±²Ό½δεζηθικλΪΫβγςσύώμνξοπρςσ£€³΄ΓΔ?Σ<’9
2’/
%"
inputs?????????Θ
p

 
ͺ "%’"

0?????????
 
%__inference_model_layer_call_fn_14343Ω~:;ΤΥΦΧΨΩΪΫXY`apq{|άέήίΰαβγ‘’±²Ό½δεζηθικλΪΫβγςσύώμνξοπρςσ£€³΄ΓΔ?Σ=’:
3’0
&#
input_1?????????Θ
p 

 
ͺ "?????????
%__inference_model_layer_call_fn_15681Ω~:;ΤΥΦΧΨΩΪΫXY`apq{|άέήίΰαβγ‘’±²Ό½δεζηθικλΪΫβγςσύώμνξοπρςσ£€³΄ΓΔ?Σ=’:
3’0
&#
input_1?????????Θ
p

 
ͺ "?????????
%__inference_model_layer_call_fn_16331Ψ~:;ΤΥΦΧΨΩΪΫXY`apq{|άέήίΰαβγ‘’±²Ό½δεζηθικλΪΫβγςσύώμνξοπρςσ£€³΄ΓΔ?Σ<’9
2’/
%"
inputs?????????Θ
p 

 
ͺ "?????????
%__inference_model_layer_call_fn_16472Ψ~:;ΤΥΦΧΨΩΪΫXY`apq{|άέήίΰαβγ‘’±²Ό½δεζηθικλΪΫβγςσύώμνξοπρςσ£€³΄ΓΔ?Σ<’9
2’/
%"
inputs?????????Θ
p

 
ͺ "??????????
Q__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_17743©άέήίΰαβγi’f
_’\
$!
query?????????Θ
$!
value?????????Θ

 

 
p 
p 
ͺ "*’'
 
0?????????Θ
 ?
Q__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_17785©άέήίΰαβγi’f
_’\
$!
query?????????Θ
$!
value?????????Θ

 

 
p 
p
ͺ "*’'
 
0?????????Θ
 Χ
6__inference_multi_head_attention_1_layer_call_fn_17686άέήίΰαβγi’f
_’\
$!
query?????????Θ
$!
value?????????Θ

 

 
p 
p 
ͺ "?????????ΘΧ
6__inference_multi_head_attention_1_layer_call_fn_17708άέήίΰαβγi’f
_’\
$!
query?????????Θ
$!
value?????????Θ

 

 
p 
p
ͺ "?????????Θ?
Q__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_18029©δεζηθικλi’f
_’\
$!
query?????????Θ
$!
value?????????Θ

 

 
p 
p 
ͺ "*’'
 
0?????????Θ
 ?
Q__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_18071©δεζηθικλi’f
_’\
$!
query?????????Θ
$!
value?????????Θ

 

 
p 
p
ͺ "*’'
 
0?????????Θ
 Χ
6__inference_multi_head_attention_2_layer_call_fn_17972δεζηθικλi’f
_’\
$!
query?????????Θ
$!
value?????????Θ

 

 
p 
p 
ͺ "?????????ΘΧ
6__inference_multi_head_attention_2_layer_call_fn_17994δεζηθικλi’f
_’\
$!
query?????????Θ
$!
value?????????Θ

 

 
p 
p
ͺ "?????????Θ?
Q__inference_multi_head_attention_3_layer_call_and_return_conditional_losses_18315©μνξοπρςσi’f
_’\
$!
query?????????Θ
$!
value?????????Θ

 

 
p 
p 
ͺ "*’'
 
0?????????Θ
 ?
Q__inference_multi_head_attention_3_layer_call_and_return_conditional_losses_18357©μνξοπρςσi’f
_’\
$!
query?????????Θ
$!
value?????????Θ

 

 
p 
p
ͺ "*’'
 
0?????????Θ
 Χ
6__inference_multi_head_attention_3_layer_call_fn_18258μνξοπρςσi’f
_’\
$!
query?????????Θ
$!
value?????????Θ

 

 
p 
p 
ͺ "?????????ΘΧ
6__inference_multi_head_attention_3_layer_call_fn_18280μνξοπρςσi’f
_’\
$!
query?????????Θ
$!
value?????????Θ

 

 
p 
p
ͺ "?????????Θύ
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_17457©ΤΥΦΧΨΩΪΫi’f
_’\
$!
query?????????Θ
$!
value?????????Θ

 

 
p 
p 
ͺ "*’'
 
0?????????Θ
 ύ
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_17499©ΤΥΦΧΨΩΪΫi’f
_’\
$!
query?????????Θ
$!
value?????????Θ

 

 
p 
p
ͺ "*’'
 
0?????????Θ
 Υ
4__inference_multi_head_attention_layer_call_fn_17400ΤΥΦΧΨΩΪΫi’f
_’\
$!
query?????????Θ
$!
value?????????Θ

 

 
p 
p 
ͺ "?????????ΘΥ
4__inference_multi_head_attention_layer_call_fn_17422ΤΥΦΧΨΩΪΫi’f
_’\
$!
query?????????Θ
$!
value?????????Θ

 

 
p 
p
ͺ "?????????Θ
#__inference_signature_wrapper_16190υ~:;ΤΥΦΧΨΩΪΫXY`apq{|άέήίΰαβγ‘’±²Ό½δεζηθικλΪΫβγςσύώμνξοπρςσ£€³΄ΓΔ?Σ@’=
’ 
6ͺ3
1
input_1&#
input_1?????????Θ"1ͺ.
,
dense_1!
dense_1?????????