Ńň
ż
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
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
­
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
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

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
Ľ
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
Á
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
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
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
-
Tanh
x"T
y"T"
Ttype:

2
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéčelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéčelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint˙˙˙˙˙˙˙˙˙
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.8.42v2.8.3-90-g1b8f5c396f08ćâ

embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	V*'
shared_nameembedding_1/embeddings

*embedding_1/embeddings/Read/ReadVariableOpReadVariableOpembedding_1/embeddings*
_output_shapes
:	V*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	V*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	V*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:V*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:V*
dtype0

gru_1/gru_cell_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_namegru_1/gru_cell_1/kernel

+gru_1/gru_cell_1/kernel/Read/ReadVariableOpReadVariableOpgru_1/gru_cell_1/kernel* 
_output_shapes
:
*
dtype0
 
!gru_1/gru_cell_1/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*2
shared_name#!gru_1/gru_cell_1/recurrent_kernel

5gru_1/gru_cell_1/recurrent_kernel/Read/ReadVariableOpReadVariableOp!gru_1/gru_cell_1/recurrent_kernel* 
_output_shapes
:
*
dtype0

gru_1/gru_cell_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namegru_1/gru_cell_1/bias
|
)gru_1/gru_cell_1/bias/Read/ReadVariableOpReadVariableOpgru_1/gru_cell_1/bias*
_output_shapes	
:*
dtype0
y
gru_1/VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namegru_1/Variable
r
"gru_1/Variable/Read/ReadVariableOpReadVariableOpgru_1/Variable*
_output_shapes
:	*
dtype0

NoOpNoOp
ş
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ő
valueëBč Bá
˛
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature

signatures*
 

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
Á
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses*
Ś

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses*
.
0
$1
%2
&3
4
5*
.
0
$1
%2
&3
4
5*
* 
°
'non_trainable_variables

(layers
)metrics
*layer_regularization_losses
+layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
* 
* 
* 

,serving_default* 
jd
VARIABLE_VALUEembedding_1/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

0*

0*
* 

-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
Ó

$kernel
%recurrent_kernel
&bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6_random_generator
7__call__
*8&call_and_return_all_conditional_losses*
* 

$0
%1
&2*

$0
%1
&2*
* 


9states
:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*
* 
* 
WQ
VARIABLE_VALUEgru_1/gru_cell_1/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!gru_1/gru_cell_1/recurrent_kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEgru_1/gru_cell_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*
* 
* 
* 
* 
* 
* 
* 
* 
* 

$0
%1
&2*

$0
%1
&2*
* 

Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
2	variables
3trainable_variables
4regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses*
* 
* 
* 

I0*
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
jd
VARIABLE_VALUEgru_1/VariableBlayer_with_weights-1/keras_api/states/0/.ATTRIBUTES/VARIABLE_VALUE*

!serving_default_embedding_1_inputPlaceholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
é
StatefulPartitionedCallStatefulPartitionedCall!serving_default_embedding_1_inputembedding_1/embeddingsgru_1/gru_cell_1/kernelgru_1/gru_cell_1/bias!gru_1/gru_cell_1/recurrent_kernelgru_1/Variabledense_1/kerneldense_1/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙V*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_291180
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ę
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*embedding_1/embeddings/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp+gru_1/gru_cell_1/kernel/Read/ReadVariableOp5gru_1/gru_cell_1/recurrent_kernel/Read/ReadVariableOp)gru_1/gru_cell_1/bias/Read/ReadVariableOp"gru_1/Variable/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_292588
š
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_1/embeddingsdense_1/kerneldense_1/biasgru_1/gru_cell_1/kernel!gru_1/gru_cell_1/recurrent_kernelgru_1/gru_cell_1/biasgru_1/Variable*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_292619Š
Ć

gru_1_while_body_290754(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2%
!gru_1_while_gru_1_strided_slice_0c
_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0D
0gru_1_while_gru_cell_1_readvariableop_resource_0:
A
2gru_1_while_gru_cell_1_readvariableop_3_resource_0:	F
2gru_1_while_gru_cell_1_readvariableop_6_resource_0:

gru_1_while_identity
gru_1_while_identity_1
gru_1_while_identity_2
gru_1_while_identity_3
gru_1_while_identity_4#
gru_1_while_gru_1_strided_slicea
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensorB
.gru_1_while_gru_cell_1_readvariableop_resource:
?
0gru_1_while_gru_cell_1_readvariableop_3_resource:	D
0gru_1_while_gru_cell_1_readvariableop_6_resource:
˘%gru_1/while/gru_cell_1/ReadVariableOp˘'gru_1/while/gru_cell_1/ReadVariableOp_1˘'gru_1/while/gru_cell_1/ReadVariableOp_2˘'gru_1/while/gru_cell_1/ReadVariableOp_3˘'gru_1/while/gru_cell_1/ReadVariableOp_4˘'gru_1/while/gru_cell_1/ReadVariableOp_5˘'gru_1/while/gru_cell_1/ReadVariableOp_6˘'gru_1/while/gru_cell_1/ReadVariableOp_7˘'gru_1/while/gru_cell_1/ReadVariableOp_8
=gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ź
/gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0gru_1_while_placeholderFgru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	*
element_dtype0
%gru_1/while/gru_cell_1/ReadVariableOpReadVariableOp0gru_1_while_gru_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0{
*gru_1/while/gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        }
,gru_1/while/gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,gru_1/while/gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ŕ
$gru_1/while/gru_cell_1/strided_sliceStridedSlice-gru_1/while/gru_cell_1/ReadVariableOp:value:03gru_1/while/gru_cell_1/strided_slice/stack:output:05gru_1/while/gru_cell_1/strided_slice/stack_1:output:05gru_1/while/gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask¸
gru_1/while/gru_cell_1/MatMulMatMul6gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0-gru_1/while/gru_cell_1/strided_slice:output:0*
T0*
_output_shapes
:	
'gru_1/while/gru_cell_1/ReadVariableOp_1ReadVariableOp0gru_1_while_gru_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0}
,gru_1/while/gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
.gru_1/while/gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.gru_1/while/gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ę
&gru_1/while/gru_cell_1/strided_slice_1StridedSlice/gru_1/while/gru_cell_1/ReadVariableOp_1:value:05gru_1/while/gru_cell_1/strided_slice_1/stack:output:07gru_1/while/gru_cell_1/strided_slice_1/stack_1:output:07gru_1/while/gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskź
gru_1/while/gru_cell_1/MatMul_1MatMul6gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0/gru_1/while/gru_cell_1/strided_slice_1:output:0*
T0*
_output_shapes
:	
'gru_1/while/gru_cell_1/ReadVariableOp_2ReadVariableOp0gru_1_while_gru_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0}
,gru_1/while/gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
.gru_1/while/gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
.gru_1/while/gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ę
&gru_1/while/gru_cell_1/strided_slice_2StridedSlice/gru_1/while/gru_cell_1/ReadVariableOp_2:value:05gru_1/while/gru_cell_1/strided_slice_2/stack:output:07gru_1/while/gru_cell_1/strided_slice_2/stack_1:output:07gru_1/while/gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskź
gru_1/while/gru_cell_1/MatMul_2MatMul6gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0/gru_1/while/gru_cell_1/strided_slice_2:output:0*
T0*
_output_shapes
:	
'gru_1/while/gru_cell_1/ReadVariableOp_3ReadVariableOp2gru_1_while_gru_cell_1_readvariableop_3_resource_0*
_output_shapes	
:*
dtype0v
,gru_1/while/gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: y
.gru_1/while/gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.gru_1/while/gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ő
&gru_1/while/gru_cell_1/strided_slice_3StridedSlice/gru_1/while/gru_cell_1/ReadVariableOp_3:value:05gru_1/while/gru_cell_1/strided_slice_3/stack:output:07gru_1/while/gru_cell_1/strided_slice_3/stack_1:output:07gru_1/while/gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*

begin_mask­
gru_1/while/gru_cell_1/BiasAddBiasAdd'gru_1/while/gru_cell_1/MatMul:product:0/gru_1/while/gru_cell_1/strided_slice_3:output:0*
T0*
_output_shapes
:	
'gru_1/while/gru_cell_1/ReadVariableOp_4ReadVariableOp2gru_1_while_gru_cell_1_readvariableop_3_resource_0*
_output_shapes	
:*
dtype0w
,gru_1/while/gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:y
.gru_1/while/gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.gru_1/while/gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ă
&gru_1/while/gru_cell_1/strided_slice_4StridedSlice/gru_1/while/gru_cell_1/ReadVariableOp_4:value:05gru_1/while/gru_cell_1/strided_slice_4/stack:output:07gru_1/while/gru_cell_1/strided_slice_4/stack_1:output:07gru_1/while/gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:ą
 gru_1/while/gru_cell_1/BiasAdd_1BiasAdd)gru_1/while/gru_cell_1/MatMul_1:product:0/gru_1/while/gru_cell_1/strided_slice_4:output:0*
T0*
_output_shapes
:	
'gru_1/while/gru_cell_1/ReadVariableOp_5ReadVariableOp2gru_1_while_gru_cell_1_readvariableop_3_resource_0*
_output_shapes	
:*
dtype0w
,gru_1/while/gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.gru_1/while/gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.gru_1/while/gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ó
&gru_1/while/gru_cell_1/strided_slice_5StridedSlice/gru_1/while/gru_cell_1/ReadVariableOp_5:value:05gru_1/while/gru_cell_1/strided_slice_5/stack:output:07gru_1/while/gru_cell_1/strided_slice_5/stack_1:output:07gru_1/while/gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*
end_maską
 gru_1/while/gru_cell_1/BiasAdd_2BiasAdd)gru_1/while/gru_cell_1/MatMul_2:product:0/gru_1/while/gru_cell_1/strided_slice_5:output:0*
T0*
_output_shapes
:	
'gru_1/while/gru_cell_1/ReadVariableOp_6ReadVariableOp2gru_1_while_gru_cell_1_readvariableop_6_resource_0* 
_output_shapes
:
*
dtype0}
,gru_1/while/gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.gru_1/while/gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.gru_1/while/gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ę
&gru_1/while/gru_cell_1/strided_slice_6StridedSlice/gru_1/while/gru_cell_1/ReadVariableOp_6:value:05gru_1/while/gru_cell_1/strided_slice_6/stack:output:07gru_1/while/gru_cell_1/strided_slice_6/stack_1:output:07gru_1/while/gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_1/while/gru_cell_1/MatMul_3MatMulgru_1_while_placeholder_2/gru_1/while/gru_cell_1/strided_slice_6:output:0*
T0*
_output_shapes
:	
'gru_1/while/gru_cell_1/ReadVariableOp_7ReadVariableOp2gru_1_while_gru_cell_1_readvariableop_6_resource_0* 
_output_shapes
:
*
dtype0}
,gru_1/while/gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       
.gru_1/while/gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.gru_1/while/gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ę
&gru_1/while/gru_cell_1/strided_slice_7StridedSlice/gru_1/while/gru_cell_1/ReadVariableOp_7:value:05gru_1/while/gru_cell_1/strided_slice_7/stack:output:07gru_1/while/gru_cell_1/strided_slice_7/stack_1:output:07gru_1/while/gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_1/while/gru_cell_1/MatMul_4MatMulgru_1_while_placeholder_2/gru_1/while/gru_cell_1/strided_slice_7:output:0*
T0*
_output_shapes
:	Ą
gru_1/while/gru_cell_1/addAddV2'gru_1/while/gru_cell_1/BiasAdd:output:0)gru_1/while/gru_cell_1/MatMul_3:product:0*
T0*
_output_shapes
:	s
gru_1/while/gru_cell_1/SigmoidSigmoidgru_1/while/gru_cell_1/add:z:0*
T0*
_output_shapes
:	Ľ
gru_1/while/gru_cell_1/add_1AddV2)gru_1/while/gru_cell_1/BiasAdd_1:output:0)gru_1/while/gru_cell_1/MatMul_4:product:0*
T0*
_output_shapes
:	w
 gru_1/while/gru_cell_1/Sigmoid_1Sigmoid gru_1/while/gru_cell_1/add_1:z:0*
T0*
_output_shapes
:	
gru_1/while/gru_cell_1/mulMul$gru_1/while/gru_cell_1/Sigmoid_1:y:0gru_1_while_placeholder_2*
T0*
_output_shapes
:	
'gru_1/while/gru_cell_1/ReadVariableOp_8ReadVariableOp2gru_1_while_gru_cell_1_readvariableop_6_resource_0* 
_output_shapes
:
*
dtype0}
,gru_1/while/gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       
.gru_1/while/gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
.gru_1/while/gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ę
&gru_1/while/gru_cell_1/strided_slice_8StridedSlice/gru_1/while/gru_cell_1/ReadVariableOp_8:value:05gru_1/while/gru_cell_1/strided_slice_8/stack:output:07gru_1/while/gru_cell_1/strided_slice_8/stack_1:output:07gru_1/while/gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask¤
gru_1/while/gru_cell_1/MatMul_5MatMulgru_1/while/gru_cell_1/mul:z:0/gru_1/while/gru_cell_1/strided_slice_8:output:0*
T0*
_output_shapes
:	Ľ
gru_1/while/gru_cell_1/add_2AddV2)gru_1/while/gru_cell_1/BiasAdd_2:output:0)gru_1/while/gru_cell_1/MatMul_5:product:0*
T0*
_output_shapes
:	o
gru_1/while/gru_cell_1/TanhTanh gru_1/while/gru_cell_1/add_2:z:0*
T0*
_output_shapes
:	
gru_1/while/gru_cell_1/mul_1Mul"gru_1/while/gru_cell_1/Sigmoid:y:0gru_1_while_placeholder_2*
T0*
_output_shapes
:	a
gru_1/while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_1/while/gru_cell_1/subSub%gru_1/while/gru_cell_1/sub/x:output:0"gru_1/while/gru_cell_1/Sigmoid:y:0*
T0*
_output_shapes
:	
gru_1/while/gru_cell_1/mul_2Mulgru_1/while/gru_cell_1/sub:z:0gru_1/while/gru_cell_1/Tanh:y:0*
T0*
_output_shapes
:	
gru_1/while/gru_cell_1/add_3AddV2 gru_1/while/gru_cell_1/mul_1:z:0 gru_1/while/gru_cell_1/mul_2:z:0*
T0*
_output_shapes
:	Ű
0gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_1_while_placeholder_1gru_1_while_placeholder gru_1/while/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:éčŇS
gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :n
gru_1/while/addAddV2gru_1_while_placeholdergru_1/while/add/y:output:0*
T0*
_output_shapes
: U
gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
gru_1/while/add_1AddV2$gru_1_while_gru_1_while_loop_countergru_1/while/add_1/y:output:0*
T0*
_output_shapes
: k
gru_1/while/IdentityIdentitygru_1/while/add_1:z:0^gru_1/while/NoOp*
T0*
_output_shapes
: 
gru_1/while/Identity_1Identity*gru_1_while_gru_1_while_maximum_iterations^gru_1/while/NoOp*
T0*
_output_shapes
: k
gru_1/while/Identity_2Identitygru_1/while/add:z:0^gru_1/while/NoOp*
T0*
_output_shapes
: Ť
gru_1/while/Identity_3Identity@gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_1/while/NoOp*
T0*
_output_shapes
: :éčŇ
gru_1/while/Identity_4Identity gru_1/while/gru_cell_1/add_3:z:0^gru_1/while/NoOp*
T0*
_output_shapes
:	Ę
gru_1/while/NoOpNoOp&^gru_1/while/gru_cell_1/ReadVariableOp(^gru_1/while/gru_cell_1/ReadVariableOp_1(^gru_1/while/gru_cell_1/ReadVariableOp_2(^gru_1/while/gru_cell_1/ReadVariableOp_3(^gru_1/while/gru_cell_1/ReadVariableOp_4(^gru_1/while/gru_cell_1/ReadVariableOp_5(^gru_1/while/gru_cell_1/ReadVariableOp_6(^gru_1/while/gru_cell_1/ReadVariableOp_7(^gru_1/while/gru_cell_1/ReadVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 "D
gru_1_while_gru_1_strided_slice!gru_1_while_gru_1_strided_slice_0"f
0gru_1_while_gru_cell_1_readvariableop_3_resource2gru_1_while_gru_cell_1_readvariableop_3_resource_0"f
0gru_1_while_gru_cell_1_readvariableop_6_resource2gru_1_while_gru_cell_1_readvariableop_6_resource_0"b
.gru_1_while_gru_cell_1_readvariableop_resource0gru_1_while_gru_cell_1_readvariableop_resource_0"5
gru_1_while_identitygru_1/while/Identity:output:0"9
gru_1_while_identity_1gru_1/while/Identity_1:output:0"9
gru_1_while_identity_2gru_1/while/Identity_2:output:0"9
gru_1_while_identity_3gru_1/while/Identity_3:output:0"9
gru_1_while_identity_4gru_1/while/Identity_4:output:0"Ŕ
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*0
_input_shapes
: : : : :	: : : : : 2N
%gru_1/while/gru_cell_1/ReadVariableOp%gru_1/while/gru_cell_1/ReadVariableOp2R
'gru_1/while/gru_cell_1/ReadVariableOp_1'gru_1/while/gru_cell_1/ReadVariableOp_12R
'gru_1/while/gru_cell_1/ReadVariableOp_2'gru_1/while/gru_cell_1/ReadVariableOp_22R
'gru_1/while/gru_cell_1/ReadVariableOp_3'gru_1/while/gru_cell_1/ReadVariableOp_32R
'gru_1/while/gru_cell_1/ReadVariableOp_4'gru_1/while/gru_cell_1/ReadVariableOp_42R
'gru_1/while/gru_cell_1/ReadVariableOp_5'gru_1/while/gru_cell_1/ReadVariableOp_52R
'gru_1/while/gru_cell_1/ReadVariableOp_6'gru_1/while/gru_cell_1/ReadVariableOp_62R
'gru_1/while/gru_cell_1/ReadVariableOp_7'gru_1/while/gru_cell_1/ReadVariableOp_72R
'gru_1/while/gru_cell_1/ReadVariableOp_8'gru_1/while/gru_cell_1/ReadVariableOp_8: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: 
ľH
˛
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_292389

inputs
states_0+
readvariableop_resource:
(
readvariableop_3_resource:	-
readvariableop_6_resource:

identity

identity_1˘ReadVariableOp˘ReadVariableOp_1˘ReadVariableOp_2˘ReadVariableOp_3˘ReadVariableOp_4˘ReadVariableOp_5˘ReadVariableOp_6˘ReadVariableOp_7˘ReadVariableOp_8h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      í
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskZ
MatMulMatMulinputsstrided_slice:output:0*
T0*
_output_shapes
:	j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask^
MatMul_1MatMulinputsstrided_slice_1:output:0*
T0*
_output_shapes
:	j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask^
MatMul_2MatMulinputsstrided_slice_2:output:0*
T0*
_output_shapes
:	g
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes	
:*
dtype0_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: b
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*

begin_maskh
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*
_output_shapes
:	g
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes	
:*
dtype0`
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Đ
strided_slice_4StridedSliceReadVariableOp_4:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:l
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*
_output_shapes
:	g
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes	
:*
dtype0`
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ŕ
strided_slice_5StridedSliceReadVariableOp_5:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*
end_maskl
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*
_output_shapes
:	l
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource* 
_output_shapes
:
*
dtype0f
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_6StridedSliceReadVariableOp_6:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask`
MatMul_3MatMulstates_0strided_slice_6:output:0*
T0*
_output_shapes
:	l
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource* 
_output_shapes
:
*
dtype0f
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_7StridedSliceReadVariableOp_7:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask`
MatMul_4MatMulstates_0strided_slice_7:output:0*
T0*
_output_shapes
:	\
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*
_output_shapes
:	E
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:	`
add_1AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*
_output_shapes
:	I
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:	M
mulMulSigmoid_1:y:0states_0*
T0*
_output_shapes
:	l
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource* 
_output_shapes
:
*
dtype0f
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_8StridedSliceReadVariableOp_8:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask_
MatMul_5MatMulmul:z:0strided_slice_8:output:0*
T0*
_output_shapes
:	`
add_2AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*
_output_shapes
:	A
TanhTanh	add_2:z:0*
T0*
_output_shapes
:	M
mul_1MulSigmoid:y:0states_0*
T0*
_output_shapes
:	J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Q
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:	I
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:	N
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:	P
IdentityIdentity	add_3:z:0^NoOp*
T0*
_output_shapes
:	R

Identity_1Identity	add_3:z:0^NoOp*
T0*
_output_shapes
:	ď
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:	:	: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_8:G C

_output_shapes
:	
 
_user_specified_nameinputs:IE

_output_shapes
:	
"
_user_specified_name
states/0
É

A__inference_gru_1_layer_call_and_return_conditional_losses_290474

inputs6
"gru_cell_1_readvariableop_resource:
3
$gru_cell_1_readvariableop_3_resource:	8
$gru_cell_1_readvariableop_6_resource:
>
+gru_cell_1_matmul_3_readvariableop_resource:	
identity˘AssignVariableOp˘ReadVariableOp˘"gru_cell_1/MatMul_3/ReadVariableOp˘"gru_cell_1/MatMul_4/ReadVariableOp˘gru_cell_1/ReadVariableOp˘gru_cell_1/ReadVariableOp_1˘gru_cell_1/ReadVariableOp_2˘gru_cell_1/ReadVariableOp_3˘gru_cell_1/ReadVariableOp_4˘gru_cell_1/ReadVariableOp_5˘gru_cell_1/ReadVariableOp_6˘gru_cell_1/ReadVariableOp_7˘gru_cell_1/ReadVariableOp_8˘gru_cell_1/mul/ReadVariableOp˘gru_cell_1/mul_1/ReadVariableOp˘whilec
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙˛
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:á
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_mask~
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype0o
gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        q
 gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       q
 gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¤
gru_cell_1/strided_sliceStridedSlice!gru_cell_1/ReadVariableOp:value:0'gru_cell_1/strided_slice/stack:output:0)gru_cell_1/strided_slice/stack_1:output:0)gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_cell_1/MatMulMatMulstrided_slice_1:output:0!gru_cell_1/strided_slice:output:0*
T0*
_output_shapes
:	
gru_cell_1/ReadVariableOp_1ReadVariableOp"gru_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype0q
 gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
gru_cell_1/strided_slice_1StridedSlice#gru_cell_1/ReadVariableOp_1:value:0)gru_cell_1/strided_slice_1/stack:output:0+gru_cell_1/strided_slice_1/stack_1:output:0+gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_cell_1/MatMul_1MatMulstrided_slice_1:output:0#gru_cell_1/strided_slice_1:output:0*
T0*
_output_shapes
:	
gru_cell_1/ReadVariableOp_2ReadVariableOp"gru_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype0q
 gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        s
"gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
gru_cell_1/strided_slice_2StridedSlice#gru_cell_1/ReadVariableOp_2:value:0)gru_cell_1/strided_slice_2/stack:output:0+gru_cell_1/strided_slice_2/stack_1:output:0+gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_cell_1/MatMul_2MatMulstrided_slice_1:output:0#gru_cell_1/strided_slice_2:output:0*
T0*
_output_shapes
:	}
gru_cell_1/ReadVariableOp_3ReadVariableOp$gru_cell_1_readvariableop_3_resource*
_output_shapes	
:*
dtype0j
 gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: m
"gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_cell_1/strided_slice_3StridedSlice#gru_cell_1/ReadVariableOp_3:value:0)gru_cell_1/strided_slice_3/stack:output:0+gru_cell_1/strided_slice_3/stack_1:output:0+gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*

begin_mask
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0#gru_cell_1/strided_slice_3:output:0*
T0*
_output_shapes
:	}
gru_cell_1/ReadVariableOp_4ReadVariableOp$gru_cell_1_readvariableop_3_resource*
_output_shapes	
:*
dtype0k
 gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:m
"gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_cell_1/strided_slice_4StridedSlice#gru_cell_1/ReadVariableOp_4:value:0)gru_cell_1/strided_slice_4/stack:output:0+gru_cell_1/strided_slice_4/stack_1:output:0+gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0#gru_cell_1/strided_slice_4:output:0*
T0*
_output_shapes
:	}
gru_cell_1/ReadVariableOp_5ReadVariableOp$gru_cell_1_readvariableop_3_resource*
_output_shapes	
:*
dtype0k
 gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:l
"gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: l
"gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_cell_1/strided_slice_5StridedSlice#gru_cell_1/ReadVariableOp_5:value:0)gru_cell_1/strided_slice_5/stack:output:0+gru_cell_1/strided_slice_5/stack_1:output:0+gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*
end_mask
gru_cell_1/BiasAdd_2BiasAddgru_cell_1/MatMul_2:product:0#gru_cell_1/strided_slice_5:output:0*
T0*
_output_shapes
:	
gru_cell_1/ReadVariableOp_6ReadVariableOp$gru_cell_1_readvariableop_6_resource* 
_output_shapes
:
*
dtype0q
 gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
gru_cell_1/strided_slice_6StridedSlice#gru_cell_1/ReadVariableOp_6:value:0)gru_cell_1/strided_slice_6/stack:output:0+gru_cell_1/strided_slice_6/stack_1:output:0+gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
"gru_cell_1/MatMul_3/ReadVariableOpReadVariableOp+gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_1/MatMul_3MatMul*gru_cell_1/MatMul_3/ReadVariableOp:value:0#gru_cell_1/strided_slice_6:output:0*
T0*
_output_shapes
:	
gru_cell_1/ReadVariableOp_7ReadVariableOp$gru_cell_1_readvariableop_6_resource* 
_output_shapes
:
*
dtype0q
 gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
gru_cell_1/strided_slice_7StridedSlice#gru_cell_1/ReadVariableOp_7:value:0)gru_cell_1/strided_slice_7/stack:output:0+gru_cell_1/strided_slice_7/stack_1:output:0+gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
"gru_cell_1/MatMul_4/ReadVariableOpReadVariableOp+gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_1/MatMul_4MatMul*gru_cell_1/MatMul_4/ReadVariableOp:value:0#gru_cell_1/strided_slice_7:output:0*
T0*
_output_shapes
:	}
gru_cell_1/addAddV2gru_cell_1/BiasAdd:output:0gru_cell_1/MatMul_3:product:0*
T0*
_output_shapes
:	[
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*
_output_shapes
:	
gru_cell_1/add_1AddV2gru_cell_1/BiasAdd_1:output:0gru_cell_1/MatMul_4:product:0*
T0*
_output_shapes
:	_
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*
_output_shapes
:	
gru_cell_1/mul/ReadVariableOpReadVariableOp+gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0%gru_cell_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:	
gru_cell_1/ReadVariableOp_8ReadVariableOp$gru_cell_1_readvariableop_6_resource* 
_output_shapes
:
*
dtype0q
 gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        s
"gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
gru_cell_1/strided_slice_8StridedSlice#gru_cell_1/ReadVariableOp_8:value:0)gru_cell_1/strided_slice_8/stack:output:0+gru_cell_1/strided_slice_8/stack_1:output:0+gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_cell_1/MatMul_5MatMulgru_cell_1/mul:z:0#gru_cell_1/strided_slice_8:output:0*
T0*
_output_shapes
:	
gru_cell_1/add_2AddV2gru_cell_1/BiasAdd_2:output:0gru_cell_1/MatMul_5:product:0*
T0*
_output_shapes
:	W
gru_cell_1/TanhTanhgru_cell_1/add_2:z:0*
T0*
_output_shapes
:	
gru_cell_1/mul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0'gru_cell_1/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	U
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?r
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*
_output_shapes
:	j
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Tanh:y:0*
T0*
_output_shapes
:	o
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*
_output_shapes
:	n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ś
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : {
ReadVariableOpReadVariableOp+gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ľ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0ReadVariableOp:value:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_1_readvariableop_resource$gru_cell_1_readvariableop_3_resource$gru_cell_1_readvariableop_6_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_290349*
condR
while_cond_290348*0
output_shapes
: : : : :	: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ă
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:˙
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    Ą
AssignVariableOpAssignVariableOp+gru_cell_1_matmul_3_readvariableop_resourcewhile:output:4^ReadVariableOp#^gru_cell_1/MatMul_3/ReadVariableOp#^gru_cell_1/MatMul_4/ReadVariableOp^gru_cell_1/mul/ReadVariableOp ^gru_cell_1/mul_1/ReadVariableOp*
_output_shapes
 *
dtype0c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp^AssignVariableOp^ReadVariableOp#^gru_cell_1/MatMul_3/ReadVariableOp#^gru_cell_1/MatMul_4/ReadVariableOp^gru_cell_1/ReadVariableOp^gru_cell_1/ReadVariableOp_1^gru_cell_1/ReadVariableOp_2^gru_cell_1/ReadVariableOp_3^gru_cell_1/ReadVariableOp_4^gru_cell_1/ReadVariableOp_5^gru_cell_1/ReadVariableOp_6^gru_cell_1/ReadVariableOp_7^gru_cell_1/ReadVariableOp_8^gru_cell_1/mul/ReadVariableOp ^gru_cell_1/mul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙: : : : 2$
AssignVariableOpAssignVariableOp2 
ReadVariableOpReadVariableOp2H
"gru_cell_1/MatMul_3/ReadVariableOp"gru_cell_1/MatMul_3/ReadVariableOp2H
"gru_cell_1/MatMul_4/ReadVariableOp"gru_cell_1/MatMul_4/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2:
gru_cell_1/ReadVariableOp_1gru_cell_1/ReadVariableOp_12:
gru_cell_1/ReadVariableOp_2gru_cell_1/ReadVariableOp_22:
gru_cell_1/ReadVariableOp_3gru_cell_1/ReadVariableOp_32:
gru_cell_1/ReadVariableOp_4gru_cell_1/ReadVariableOp_42:
gru_cell_1/ReadVariableOp_5gru_cell_1/ReadVariableOp_52:
gru_cell_1/ReadVariableOp_6gru_cell_1/ReadVariableOp_62:
gru_cell_1/ReadVariableOp_7gru_cell_1/ReadVariableOp_72:
gru_cell_1/ReadVariableOp_8gru_cell_1/ReadVariableOp_82>
gru_cell_1/mul/ReadVariableOpgru_cell_1/mul/ReadVariableOp2B
gru_cell_1/mul_1/ReadVariableOpgru_cell_1/mul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ć
¨
while_cond_291567
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice4
0while_while_cond_291567___redundant_placeholder04
0while_while_cond_291567___redundant_placeholder14
0while_while_cond_291567___redundant_placeholder24
0while_while_cond_291567___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :	: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
:
çv
Ť	
while_body_290038
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
*while_gru_cell_1_readvariableop_resource_0:
;
,while_gru_cell_1_readvariableop_3_resource_0:	@
,while_gru_cell_1_readvariableop_6_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
(while_gru_cell_1_readvariableop_resource:
9
*while_gru_cell_1_readvariableop_3_resource:	>
*while_gru_cell_1_readvariableop_6_resource:
˘while/gru_cell_1/ReadVariableOp˘!while/gru_cell_1/ReadVariableOp_1˘!while/gru_cell_1/ReadVariableOp_2˘!while/gru_cell_1/ReadVariableOp_3˘!while/gru_cell_1/ReadVariableOp_4˘!while/gru_cell_1/ReadVariableOp_5˘!while/gru_cell_1/ReadVariableOp_6˘!while/gru_cell_1/ReadVariableOp_7˘!while/gru_cell_1/ReadVariableOp_8
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	*
element_dtype0
while/gru_cell_1/ReadVariableOpReadVariableOp*while_gru_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0u
$while/gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        w
&while/gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       w
&while/gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Â
while/gru_cell_1/strided_sliceStridedSlice'while/gru_cell_1/ReadVariableOp:value:0-while/gru_cell_1/strided_slice/stack:output:0/while/gru_cell_1/strided_slice/stack_1:output:0/while/gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskŚ
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell_1/strided_slice:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_1ReadVariableOp*while_gru_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0w
&while/gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/gru_cell_1/strided_slice_1StridedSlice)while/gru_cell_1/ReadVariableOp_1:value:0/while/gru_cell_1/strided_slice_1/stack:output:01while/gru_cell_1/strided_slice_1/stack_1:output:01while/gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskŞ
while/gru_cell_1/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_1/strided_slice_1:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_2ReadVariableOp*while_gru_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0w
&while/gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        y
(while/gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/gru_cell_1/strided_slice_2StridedSlice)while/gru_cell_1/ReadVariableOp_2:value:0/while/gru_cell_1/strided_slice_2/stack:output:01while/gru_cell_1/strided_slice_2/stack_1:output:01while/gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskŞ
while/gru_cell_1/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_1/strided_slice_2:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_3ReadVariableOp,while_gru_cell_1_readvariableop_3_resource_0*
_output_shapes	
:*
dtype0p
&while/gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: s
(while/gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(while/gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ˇ
 while/gru_cell_1/strided_slice_3StridedSlice)while/gru_cell_1/ReadVariableOp_3:value:0/while/gru_cell_1/strided_slice_3/stack:output:01while/gru_cell_1/strided_slice_3/stack_1:output:01while/gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*

begin_mask
while/gru_cell_1/BiasAddBiasAdd!while/gru_cell_1/MatMul:product:0)while/gru_cell_1/strided_slice_3:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_4ReadVariableOp,while_gru_cell_1_readvariableop_3_resource_0*
_output_shapes	
:*
dtype0q
&while/gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:s
(while/gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(while/gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ľ
 while/gru_cell_1/strided_slice_4StridedSlice)while/gru_cell_1/ReadVariableOp_4:value:0/while/gru_cell_1/strided_slice_4/stack:output:01while/gru_cell_1/strided_slice_4/stack_1:output:01while/gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:
while/gru_cell_1/BiasAdd_1BiasAdd#while/gru_cell_1/MatMul_1:product:0)while/gru_cell_1/strided_slice_4:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_5ReadVariableOp,while_gru_cell_1_readvariableop_3_resource_0*
_output_shapes	
:*
dtype0q
&while/gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(while/gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(while/gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ľ
 while/gru_cell_1/strided_slice_5StridedSlice)while/gru_cell_1/ReadVariableOp_5:value:0/while/gru_cell_1/strided_slice_5/stack:output:01while/gru_cell_1/strided_slice_5/stack_1:output:01while/gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*
end_mask
while/gru_cell_1/BiasAdd_2BiasAdd#while/gru_cell_1/MatMul_2:product:0)while/gru_cell_1/strided_slice_5:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_6ReadVariableOp,while_gru_cell_1_readvariableop_6_resource_0* 
_output_shapes
:
*
dtype0w
&while/gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/gru_cell_1/strided_slice_6StridedSlice)while/gru_cell_1/ReadVariableOp_6:value:0/while/gru_cell_1/strided_slice_6/stack:output:01while/gru_cell_1/strided_slice_6/stack_1:output:01while/gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/gru_cell_1/MatMul_3MatMulwhile_placeholder_2)while/gru_cell_1/strided_slice_6:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_7ReadVariableOp,while_gru_cell_1_readvariableop_6_resource_0* 
_output_shapes
:
*
dtype0w
&while/gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/gru_cell_1/strided_slice_7StridedSlice)while/gru_cell_1/ReadVariableOp_7:value:0/while/gru_cell_1/strided_slice_7/stack:output:01while/gru_cell_1/strided_slice_7/stack_1:output:01while/gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/gru_cell_1/MatMul_4MatMulwhile_placeholder_2)while/gru_cell_1/strided_slice_7:output:0*
T0*
_output_shapes
:	
while/gru_cell_1/addAddV2!while/gru_cell_1/BiasAdd:output:0#while/gru_cell_1/MatMul_3:product:0*
T0*
_output_shapes
:	g
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*
_output_shapes
:	
while/gru_cell_1/add_1AddV2#while/gru_cell_1/BiasAdd_1:output:0#while/gru_cell_1/MatMul_4:product:0*
T0*
_output_shapes
:	k
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*
_output_shapes
:	z
while/gru_cell_1/mulMulwhile/gru_cell_1/Sigmoid_1:y:0while_placeholder_2*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_8ReadVariableOp,while_gru_cell_1_readvariableop_6_resource_0* 
_output_shapes
:
*
dtype0w
&while/gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        y
(while/gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/gru_cell_1/strided_slice_8StridedSlice)while/gru_cell_1/ReadVariableOp_8:value:0/while/gru_cell_1/strided_slice_8/stack:output:01while/gru_cell_1/strided_slice_8/stack_1:output:01while/gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/gru_cell_1/MatMul_5MatMulwhile/gru_cell_1/mul:z:0)while/gru_cell_1/strided_slice_8:output:0*
T0*
_output_shapes
:	
while/gru_cell_1/add_2AddV2#while/gru_cell_1/BiasAdd_2:output:0#while/gru_cell_1/MatMul_5:product:0*
T0*
_output_shapes
:	c
while/gru_cell_1/TanhTanhwhile/gru_cell_1/add_2:z:0*
T0*
_output_shapes
:	z
while/gru_cell_1/mul_1Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*
_output_shapes
:	[
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*
_output_shapes
:	|
while/gru_cell_1/mul_2Mulwhile/gru_cell_1/sub:z:0while/gru_cell_1/Tanh:y:0*
T0*
_output_shapes
:	
while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_1:z:0while/gru_cell_1/mul_2:z:0*
T0*
_output_shapes
:	Ă
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:éčŇM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éčŇo
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0^while/NoOp*
T0*
_output_shapes
:	

while/NoOpNoOp ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6"^while/gru_cell_1/ReadVariableOp_7"^while/gru_cell_1/ReadVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 "Z
*while_gru_cell_1_readvariableop_3_resource,while_gru_cell_1_readvariableop_3_resource_0"Z
*while_gru_cell_1_readvariableop_6_resource,while_gru_cell_1_readvariableop_6_resource_0"V
(while_gru_cell_1_readvariableop_resource*while_gru_cell_1_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*0
_input_shapes
: : : : :	: : : : : 2B
while/gru_cell_1/ReadVariableOpwhile/gru_cell_1/ReadVariableOp2F
!while/gru_cell_1/ReadVariableOp_1!while/gru_cell_1/ReadVariableOp_12F
!while/gru_cell_1/ReadVariableOp_2!while/gru_cell_1/ReadVariableOp_22F
!while/gru_cell_1/ReadVariableOp_3!while/gru_cell_1/ReadVariableOp_32F
!while/gru_cell_1/ReadVariableOp_4!while/gru_cell_1/ReadVariableOp_42F
!while/gru_cell_1/ReadVariableOp_5!while/gru_cell_1/ReadVariableOp_52F
!while/gru_cell_1/ReadVariableOp_6!while/gru_cell_1/ReadVariableOp_62F
!while/gru_cell_1/ReadVariableOp_7!while/gru_cell_1/ReadVariableOp_72F
!while/gru_cell_1/ReadVariableOp_8!while/gru_cell_1/ReadVariableOp_8: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: 
ť
	
H__inference_sequential_1_layer_call_and_return_conditional_losses_290905

inputs6
#embedding_1_embedding_lookup_290655:	V<
(gru_1_gru_cell_1_readvariableop_resource:
9
*gru_1_gru_cell_1_readvariableop_3_resource:	>
*gru_1_gru_cell_1_readvariableop_6_resource:
D
1gru_1_gru_cell_1_matmul_3_readvariableop_resource:	<
)dense_1_tensordot_readvariableop_resource:	V5
'dense_1_biasadd_readvariableop_resource:V
identity˘dense_1/BiasAdd/ReadVariableOp˘ dense_1/Tensordot/ReadVariableOp˘embedding_1/embedding_lookup˘gru_1/AssignVariableOp˘gru_1/ReadVariableOp˘(gru_1/gru_cell_1/MatMul_3/ReadVariableOp˘(gru_1/gru_cell_1/MatMul_4/ReadVariableOp˘gru_1/gru_cell_1/ReadVariableOp˘!gru_1/gru_cell_1/ReadVariableOp_1˘!gru_1/gru_cell_1/ReadVariableOp_2˘!gru_1/gru_cell_1/ReadVariableOp_3˘!gru_1/gru_cell_1/ReadVariableOp_4˘!gru_1/gru_cell_1/ReadVariableOp_5˘!gru_1/gru_cell_1/ReadVariableOp_6˘!gru_1/gru_cell_1/ReadVariableOp_7˘!gru_1/gru_cell_1/ReadVariableOp_8˘#gru_1/gru_cell_1/mul/ReadVariableOp˘%gru_1/gru_cell_1/mul_1/ReadVariableOp˘gru_1/whilea
embedding_1/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ě
embedding_1/embedding_lookupResourceGather#embedding_1_embedding_lookup_290655embedding_1/Cast:y:0*
Tindices0*6
_class,
*(loc:@embedding_1/embedding_lookup/290655*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0Ç
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_1/embedding_lookup/290655*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙i
gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¤
gru_1/transpose	Transpose0embedding_1/embedding_lookup/Identity_1:output:0gru_1/transpose/perm:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙N
gru_1/ShapeShapegru_1/transpose:y:0*
T0*
_output_shapes
:c
gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ď
gru_1/strided_sliceStridedSlicegru_1/Shape:output:0"gru_1/strided_slice/stack:output:0$gru_1/strided_slice/stack_1:output:0$gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ä
gru_1/TensorArrayV2TensorListReserve*gru_1/TensorArrayV2/element_shape:output:0gru_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
;gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ň
-gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_1/transpose:y:0Dgru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇe
gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:˙
gru_1/strided_slice_1StridedSlicegru_1/transpose:y:0$gru_1/strided_slice_1/stack:output:0&gru_1/strided_slice_1/stack_1:output:0&gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_mask
gru_1/gru_cell_1/ReadVariableOpReadVariableOp(gru_1_gru_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype0u
$gru_1/gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        w
&gru_1/gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       w
&gru_1/gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Â
gru_1/gru_cell_1/strided_sliceStridedSlice'gru_1/gru_cell_1/ReadVariableOp:value:0-gru_1/gru_cell_1/strided_slice/stack:output:0/gru_1/gru_cell_1/strided_slice/stack_1:output:0/gru_1/gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_1/gru_cell_1/MatMulMatMulgru_1/strided_slice_1:output:0'gru_1/gru_cell_1/strided_slice:output:0*
T0*
_output_shapes
:	
!gru_1/gru_cell_1/ReadVariableOp_1ReadVariableOp(gru_1_gru_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype0w
&gru_1/gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       y
(gru_1/gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(gru_1/gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 gru_1/gru_cell_1/strided_slice_1StridedSlice)gru_1/gru_cell_1/ReadVariableOp_1:value:0/gru_1/gru_cell_1/strided_slice_1/stack:output:01gru_1/gru_cell_1/strided_slice_1/stack_1:output:01gru_1/gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_1/gru_cell_1/MatMul_1MatMulgru_1/strided_slice_1:output:0)gru_1/gru_cell_1/strided_slice_1:output:0*
T0*
_output_shapes
:	
!gru_1/gru_cell_1/ReadVariableOp_2ReadVariableOp(gru_1_gru_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype0w
&gru_1/gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       y
(gru_1/gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        y
(gru_1/gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 gru_1/gru_cell_1/strided_slice_2StridedSlice)gru_1/gru_cell_1/ReadVariableOp_2:value:0/gru_1/gru_cell_1/strided_slice_2/stack:output:01gru_1/gru_cell_1/strided_slice_2/stack_1:output:01gru_1/gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_1/gru_cell_1/MatMul_2MatMulgru_1/strided_slice_1:output:0)gru_1/gru_cell_1/strided_slice_2:output:0*
T0*
_output_shapes
:	
!gru_1/gru_cell_1/ReadVariableOp_3ReadVariableOp*gru_1_gru_cell_1_readvariableop_3_resource*
_output_shapes	
:*
dtype0p
&gru_1/gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: s
(gru_1/gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(gru_1/gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ˇ
 gru_1/gru_cell_1/strided_slice_3StridedSlice)gru_1/gru_cell_1/ReadVariableOp_3:value:0/gru_1/gru_cell_1/strided_slice_3/stack:output:01gru_1/gru_cell_1/strided_slice_3/stack_1:output:01gru_1/gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*

begin_mask
gru_1/gru_cell_1/BiasAddBiasAdd!gru_1/gru_cell_1/MatMul:product:0)gru_1/gru_cell_1/strided_slice_3:output:0*
T0*
_output_shapes
:	
!gru_1/gru_cell_1/ReadVariableOp_4ReadVariableOp*gru_1_gru_cell_1_readvariableop_3_resource*
_output_shapes	
:*
dtype0q
&gru_1/gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:s
(gru_1/gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(gru_1/gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ľ
 gru_1/gru_cell_1/strided_slice_4StridedSlice)gru_1/gru_cell_1/ReadVariableOp_4:value:0/gru_1/gru_cell_1/strided_slice_4/stack:output:01gru_1/gru_cell_1/strided_slice_4/stack_1:output:01gru_1/gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:
gru_1/gru_cell_1/BiasAdd_1BiasAdd#gru_1/gru_cell_1/MatMul_1:product:0)gru_1/gru_cell_1/strided_slice_4:output:0*
T0*
_output_shapes
:	
!gru_1/gru_cell_1/ReadVariableOp_5ReadVariableOp*gru_1_gru_cell_1_readvariableop_3_resource*
_output_shapes	
:*
dtype0q
&gru_1/gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(gru_1/gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(gru_1/gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ľ
 gru_1/gru_cell_1/strided_slice_5StridedSlice)gru_1/gru_cell_1/ReadVariableOp_5:value:0/gru_1/gru_cell_1/strided_slice_5/stack:output:01gru_1/gru_cell_1/strided_slice_5/stack_1:output:01gru_1/gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*
end_mask
gru_1/gru_cell_1/BiasAdd_2BiasAdd#gru_1/gru_cell_1/MatMul_2:product:0)gru_1/gru_cell_1/strided_slice_5:output:0*
T0*
_output_shapes
:	
!gru_1/gru_cell_1/ReadVariableOp_6ReadVariableOp*gru_1_gru_cell_1_readvariableop_6_resource* 
_output_shapes
:
*
dtype0w
&gru_1/gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(gru_1/gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(gru_1/gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 gru_1/gru_cell_1/strided_slice_6StridedSlice)gru_1/gru_cell_1/ReadVariableOp_6:value:0/gru_1/gru_cell_1/strided_slice_6/stack:output:01gru_1/gru_cell_1/strided_slice_6/stack_1:output:01gru_1/gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
(gru_1/gru_cell_1/MatMul_3/ReadVariableOpReadVariableOp1gru_1_gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0Ş
gru_1/gru_cell_1/MatMul_3MatMul0gru_1/gru_cell_1/MatMul_3/ReadVariableOp:value:0)gru_1/gru_cell_1/strided_slice_6:output:0*
T0*
_output_shapes
:	
!gru_1/gru_cell_1/ReadVariableOp_7ReadVariableOp*gru_1_gru_cell_1_readvariableop_6_resource* 
_output_shapes
:
*
dtype0w
&gru_1/gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       y
(gru_1/gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(gru_1/gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 gru_1/gru_cell_1/strided_slice_7StridedSlice)gru_1/gru_cell_1/ReadVariableOp_7:value:0/gru_1/gru_cell_1/strided_slice_7/stack:output:01gru_1/gru_cell_1/strided_slice_7/stack_1:output:01gru_1/gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
(gru_1/gru_cell_1/MatMul_4/ReadVariableOpReadVariableOp1gru_1_gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0Ş
gru_1/gru_cell_1/MatMul_4MatMul0gru_1/gru_cell_1/MatMul_4/ReadVariableOp:value:0)gru_1/gru_cell_1/strided_slice_7:output:0*
T0*
_output_shapes
:	
gru_1/gru_cell_1/addAddV2!gru_1/gru_cell_1/BiasAdd:output:0#gru_1/gru_cell_1/MatMul_3:product:0*
T0*
_output_shapes
:	g
gru_1/gru_cell_1/SigmoidSigmoidgru_1/gru_cell_1/add:z:0*
T0*
_output_shapes
:	
gru_1/gru_cell_1/add_1AddV2#gru_1/gru_cell_1/BiasAdd_1:output:0#gru_1/gru_cell_1/MatMul_4:product:0*
T0*
_output_shapes
:	k
gru_1/gru_cell_1/Sigmoid_1Sigmoidgru_1/gru_cell_1/add_1:z:0*
T0*
_output_shapes
:	
#gru_1/gru_cell_1/mul/ReadVariableOpReadVariableOp1gru_1_gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_1/gru_cell_1/mulMulgru_1/gru_cell_1/Sigmoid_1:y:0+gru_1/gru_cell_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:	
!gru_1/gru_cell_1/ReadVariableOp_8ReadVariableOp*gru_1_gru_cell_1_readvariableop_6_resource* 
_output_shapes
:
*
dtype0w
&gru_1/gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       y
(gru_1/gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        y
(gru_1/gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 gru_1/gru_cell_1/strided_slice_8StridedSlice)gru_1/gru_cell_1/ReadVariableOp_8:value:0/gru_1/gru_cell_1/strided_slice_8/stack:output:01gru_1/gru_cell_1/strided_slice_8/stack_1:output:01gru_1/gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_1/gru_cell_1/MatMul_5MatMulgru_1/gru_cell_1/mul:z:0)gru_1/gru_cell_1/strided_slice_8:output:0*
T0*
_output_shapes
:	
gru_1/gru_cell_1/add_2AddV2#gru_1/gru_cell_1/BiasAdd_2:output:0#gru_1/gru_cell_1/MatMul_5:product:0*
T0*
_output_shapes
:	c
gru_1/gru_cell_1/TanhTanhgru_1/gru_cell_1/add_2:z:0*
T0*
_output_shapes
:	
%gru_1/gru_cell_1/mul_1/ReadVariableOpReadVariableOp1gru_1_gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_1/gru_cell_1/mul_1Mulgru_1/gru_cell_1/Sigmoid:y:0-gru_1/gru_cell_1/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	[
gru_1/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_1/gru_cell_1/subSubgru_1/gru_cell_1/sub/x:output:0gru_1/gru_cell_1/Sigmoid:y:0*
T0*
_output_shapes
:	|
gru_1/gru_cell_1/mul_2Mulgru_1/gru_cell_1/sub:z:0gru_1/gru_cell_1/Tanh:y:0*
T0*
_output_shapes
:	
gru_1/gru_cell_1/add_3AddV2gru_1/gru_cell_1/mul_1:z:0gru_1/gru_cell_1/mul_2:z:0*
T0*
_output_shapes
:	t
#gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      Č
gru_1/TensorArrayV2_1TensorListReserve,gru_1/TensorArrayV2_1/element_shape:output:0gru_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇL

gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 
gru_1/ReadVariableOpReadVariableOp1gru_1_gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0i
gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Z
gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ó
gru_1/whileWhile!gru_1/while/loop_counter:output:0'gru_1/while/maximum_iterations:output:0gru_1/time:output:0gru_1/TensorArrayV2_1:handle:0gru_1/ReadVariableOp:value:0gru_1/strided_slice:output:0=gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0(gru_1_gru_cell_1_readvariableop_resource*gru_1_gru_cell_1_readvariableop_3_resource*gru_1_gru_cell_1_readvariableop_6_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *#
bodyR
gru_1_while_body_290754*#
condR
gru_1_while_cond_290753*0
output_shapes
: : : : :	: : : : : *
parallel_iterations 
6gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ő
(gru_1/TensorArrayV2Stack/TensorListStackTensorListStackgru_1/while:output:3?gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0n
gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙g
gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_1/strided_slice_2StridedSlice1gru_1/TensorArrayV2Stack/TensorListStack:tensor:0$gru_1/strided_slice_2/stack:output:0&gru_1/strided_slice_2/stack_1:output:0&gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_maskk
gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Š
gru_1/transpose_1	Transpose1gru_1/TensorArrayV2Stack/TensorListStack:tensor:0gru_1/transpose_1/perm:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙a
gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    Ń
gru_1/AssignVariableOpAssignVariableOp1gru_1_gru_cell_1_matmul_3_readvariableop_resourcegru_1/while:output:4^gru_1/ReadVariableOp)^gru_1/gru_cell_1/MatMul_3/ReadVariableOp)^gru_1/gru_cell_1/MatMul_4/ReadVariableOp$^gru_1/gru_cell_1/mul/ReadVariableOp&^gru_1/gru_cell_1/mul_1/ReadVariableOp*
_output_shapes
 *
dtype0
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes
:	V*
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       \
dense_1/Tensordot/ShapeShapegru_1/transpose_1:y:0*
T0*
_output_shapes
:a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ű
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ź
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_1/Tensordot/transpose	Transposegru_1/transpose_1:y:0!dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˘
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Vc
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Va
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙V
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:V*
dtype0
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙Vk
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙VÍ
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp^embedding_1/embedding_lookup^gru_1/AssignVariableOp^gru_1/ReadVariableOp)^gru_1/gru_cell_1/MatMul_3/ReadVariableOp)^gru_1/gru_cell_1/MatMul_4/ReadVariableOp ^gru_1/gru_cell_1/ReadVariableOp"^gru_1/gru_cell_1/ReadVariableOp_1"^gru_1/gru_cell_1/ReadVariableOp_2"^gru_1/gru_cell_1/ReadVariableOp_3"^gru_1/gru_cell_1/ReadVariableOp_4"^gru_1/gru_cell_1/ReadVariableOp_5"^gru_1/gru_cell_1/ReadVariableOp_6"^gru_1/gru_cell_1/ReadVariableOp_7"^gru_1/gru_cell_1/ReadVariableOp_8$^gru_1/gru_cell_1/mul/ReadVariableOp&^gru_1/gru_cell_1/mul_1/ReadVariableOp^gru_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:˙˙˙˙˙˙˙˙˙: : : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup20
gru_1/AssignVariableOpgru_1/AssignVariableOp2,
gru_1/ReadVariableOpgru_1/ReadVariableOp2T
(gru_1/gru_cell_1/MatMul_3/ReadVariableOp(gru_1/gru_cell_1/MatMul_3/ReadVariableOp2T
(gru_1/gru_cell_1/MatMul_4/ReadVariableOp(gru_1/gru_cell_1/MatMul_4/ReadVariableOp2B
gru_1/gru_cell_1/ReadVariableOpgru_1/gru_cell_1/ReadVariableOp2F
!gru_1/gru_cell_1/ReadVariableOp_1!gru_1/gru_cell_1/ReadVariableOp_12F
!gru_1/gru_cell_1/ReadVariableOp_2!gru_1/gru_cell_1/ReadVariableOp_22F
!gru_1/gru_cell_1/ReadVariableOp_3!gru_1/gru_cell_1/ReadVariableOp_32F
!gru_1/gru_cell_1/ReadVariableOp_4!gru_1/gru_cell_1/ReadVariableOp_42F
!gru_1/gru_cell_1/ReadVariableOp_5!gru_1/gru_cell_1/ReadVariableOp_52F
!gru_1/gru_cell_1/ReadVariableOp_6!gru_1/gru_cell_1/ReadVariableOp_62F
!gru_1/gru_cell_1/ReadVariableOp_7!gru_1/gru_cell_1/ReadVariableOp_72F
!gru_1/gru_cell_1/ReadVariableOp_8!gru_1/gru_cell_1/ReadVariableOp_82J
#gru_1/gru_cell_1/mul/ReadVariableOp#gru_1/gru_cell_1/mul/ReadVariableOp2N
%gru_1/gru_cell_1/mul_1/ReadVariableOp%gru_1/gru_cell_1/mul_1/ReadVariableOp2
gru_1/whilegru_1/while:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
č

gru_1_while_cond_290753(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2(
$gru_1_while_less_gru_1_strided_slice@
<gru_1_while_gru_1_while_cond_290753___redundant_placeholder0@
<gru_1_while_gru_1_while_cond_290753___redundant_placeholder1@
<gru_1_while_gru_1_while_cond_290753___redundant_placeholder2@
<gru_1_while_gru_1_while_cond_290753___redundant_placeholder3
gru_1_while_identity
x
gru_1/while/LessLessgru_1_while_placeholder$gru_1_while_less_gru_1_strided_slice*
T0*
_output_shapes
: W
gru_1/while/IdentityIdentitygru_1/while/Less:z:0*
T0
*
_output_shapes
: "5
gru_1_while_identitygru_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :	: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
:
Ž

,__inference_embedding_1_layer_call_fn_291187

inputs
unknown:	V
identity˘StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_embedding_1_layer_call_and_return_conditional_losses_289938t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:˙˙˙˙˙˙˙˙˙: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
°
Ó
&__inference_gru_1_layer_call_fn_291249

inputs
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
identity˘StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_290474t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ŠH
°
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_289691

inputs

states+
readvariableop_resource:
(
readvariableop_3_resource:	-
readvariableop_6_resource:

identity

identity_1˘ReadVariableOp˘ReadVariableOp_1˘ReadVariableOp_2˘ReadVariableOp_3˘ReadVariableOp_4˘ReadVariableOp_5˘ReadVariableOp_6˘ReadVariableOp_7˘ReadVariableOp_8h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      í
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskZ
MatMulMatMulinputsstrided_slice:output:0*
T0*
_output_shapes
:	j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask^
MatMul_1MatMulinputsstrided_slice_1:output:0*
T0*
_output_shapes
:	j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask^
MatMul_2MatMulinputsstrided_slice_2:output:0*
T0*
_output_shapes
:	g
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes	
:*
dtype0_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: b
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*

begin_maskh
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*
_output_shapes
:	g
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes	
:*
dtype0`
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Đ
strided_slice_4StridedSliceReadVariableOp_4:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:l
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*
_output_shapes
:	g
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes	
:*
dtype0`
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ŕ
strided_slice_5StridedSliceReadVariableOp_5:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*
end_maskl
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*
_output_shapes
:	l
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource* 
_output_shapes
:
*
dtype0f
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_6StridedSliceReadVariableOp_6:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask^
MatMul_3MatMulstatesstrided_slice_6:output:0*
T0*
_output_shapes
:	l
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource* 
_output_shapes
:
*
dtype0f
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_7StridedSliceReadVariableOp_7:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask^
MatMul_4MatMulstatesstrided_slice_7:output:0*
T0*
_output_shapes
:	\
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*
_output_shapes
:	E
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:	`
add_1AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*
_output_shapes
:	I
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:	K
mulMulSigmoid_1:y:0states*
T0*
_output_shapes
:	l
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource* 
_output_shapes
:
*
dtype0f
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_8StridedSliceReadVariableOp_8:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask_
MatMul_5MatMulmul:z:0strided_slice_8:output:0*
T0*
_output_shapes
:	`
add_2AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*
_output_shapes
:	A
TanhTanh	add_2:z:0*
T0*
_output_shapes
:	K
mul_1MulSigmoid:y:0states*
T0*
_output_shapes
:	J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Q
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:	I
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:	N
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:	P
IdentityIdentity	add_3:z:0^NoOp*
T0*
_output_shapes
:	R

Identity_1Identity	add_3:z:0^NoOp*
T0*
_output_shapes
:	ď
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:	:	: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_8:G C

_output_shapes
:	
 
_user_specified_nameinputs:GC

_output_shapes
:	
 
_user_specified_namestates
çv
Ť	
while_body_291568
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
*while_gru_cell_1_readvariableop_resource_0:
;
,while_gru_cell_1_readvariableop_3_resource_0:	@
,while_gru_cell_1_readvariableop_6_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
(while_gru_cell_1_readvariableop_resource:
9
*while_gru_cell_1_readvariableop_3_resource:	>
*while_gru_cell_1_readvariableop_6_resource:
˘while/gru_cell_1/ReadVariableOp˘!while/gru_cell_1/ReadVariableOp_1˘!while/gru_cell_1/ReadVariableOp_2˘!while/gru_cell_1/ReadVariableOp_3˘!while/gru_cell_1/ReadVariableOp_4˘!while/gru_cell_1/ReadVariableOp_5˘!while/gru_cell_1/ReadVariableOp_6˘!while/gru_cell_1/ReadVariableOp_7˘!while/gru_cell_1/ReadVariableOp_8
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	*
element_dtype0
while/gru_cell_1/ReadVariableOpReadVariableOp*while_gru_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0u
$while/gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        w
&while/gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       w
&while/gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Â
while/gru_cell_1/strided_sliceStridedSlice'while/gru_cell_1/ReadVariableOp:value:0-while/gru_cell_1/strided_slice/stack:output:0/while/gru_cell_1/strided_slice/stack_1:output:0/while/gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskŚ
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell_1/strided_slice:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_1ReadVariableOp*while_gru_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0w
&while/gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/gru_cell_1/strided_slice_1StridedSlice)while/gru_cell_1/ReadVariableOp_1:value:0/while/gru_cell_1/strided_slice_1/stack:output:01while/gru_cell_1/strided_slice_1/stack_1:output:01while/gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskŞ
while/gru_cell_1/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_1/strided_slice_1:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_2ReadVariableOp*while_gru_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0w
&while/gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        y
(while/gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/gru_cell_1/strided_slice_2StridedSlice)while/gru_cell_1/ReadVariableOp_2:value:0/while/gru_cell_1/strided_slice_2/stack:output:01while/gru_cell_1/strided_slice_2/stack_1:output:01while/gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskŞ
while/gru_cell_1/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_1/strided_slice_2:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_3ReadVariableOp,while_gru_cell_1_readvariableop_3_resource_0*
_output_shapes	
:*
dtype0p
&while/gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: s
(while/gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(while/gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ˇ
 while/gru_cell_1/strided_slice_3StridedSlice)while/gru_cell_1/ReadVariableOp_3:value:0/while/gru_cell_1/strided_slice_3/stack:output:01while/gru_cell_1/strided_slice_3/stack_1:output:01while/gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*

begin_mask
while/gru_cell_1/BiasAddBiasAdd!while/gru_cell_1/MatMul:product:0)while/gru_cell_1/strided_slice_3:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_4ReadVariableOp,while_gru_cell_1_readvariableop_3_resource_0*
_output_shapes	
:*
dtype0q
&while/gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:s
(while/gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(while/gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ľ
 while/gru_cell_1/strided_slice_4StridedSlice)while/gru_cell_1/ReadVariableOp_4:value:0/while/gru_cell_1/strided_slice_4/stack:output:01while/gru_cell_1/strided_slice_4/stack_1:output:01while/gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:
while/gru_cell_1/BiasAdd_1BiasAdd#while/gru_cell_1/MatMul_1:product:0)while/gru_cell_1/strided_slice_4:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_5ReadVariableOp,while_gru_cell_1_readvariableop_3_resource_0*
_output_shapes	
:*
dtype0q
&while/gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(while/gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(while/gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ľ
 while/gru_cell_1/strided_slice_5StridedSlice)while/gru_cell_1/ReadVariableOp_5:value:0/while/gru_cell_1/strided_slice_5/stack:output:01while/gru_cell_1/strided_slice_5/stack_1:output:01while/gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*
end_mask
while/gru_cell_1/BiasAdd_2BiasAdd#while/gru_cell_1/MatMul_2:product:0)while/gru_cell_1/strided_slice_5:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_6ReadVariableOp,while_gru_cell_1_readvariableop_6_resource_0* 
_output_shapes
:
*
dtype0w
&while/gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/gru_cell_1/strided_slice_6StridedSlice)while/gru_cell_1/ReadVariableOp_6:value:0/while/gru_cell_1/strided_slice_6/stack:output:01while/gru_cell_1/strided_slice_6/stack_1:output:01while/gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/gru_cell_1/MatMul_3MatMulwhile_placeholder_2)while/gru_cell_1/strided_slice_6:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_7ReadVariableOp,while_gru_cell_1_readvariableop_6_resource_0* 
_output_shapes
:
*
dtype0w
&while/gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/gru_cell_1/strided_slice_7StridedSlice)while/gru_cell_1/ReadVariableOp_7:value:0/while/gru_cell_1/strided_slice_7/stack:output:01while/gru_cell_1/strided_slice_7/stack_1:output:01while/gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/gru_cell_1/MatMul_4MatMulwhile_placeholder_2)while/gru_cell_1/strided_slice_7:output:0*
T0*
_output_shapes
:	
while/gru_cell_1/addAddV2!while/gru_cell_1/BiasAdd:output:0#while/gru_cell_1/MatMul_3:product:0*
T0*
_output_shapes
:	g
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*
_output_shapes
:	
while/gru_cell_1/add_1AddV2#while/gru_cell_1/BiasAdd_1:output:0#while/gru_cell_1/MatMul_4:product:0*
T0*
_output_shapes
:	k
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*
_output_shapes
:	z
while/gru_cell_1/mulMulwhile/gru_cell_1/Sigmoid_1:y:0while_placeholder_2*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_8ReadVariableOp,while_gru_cell_1_readvariableop_6_resource_0* 
_output_shapes
:
*
dtype0w
&while/gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        y
(while/gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/gru_cell_1/strided_slice_8StridedSlice)while/gru_cell_1/ReadVariableOp_8:value:0/while/gru_cell_1/strided_slice_8/stack:output:01while/gru_cell_1/strided_slice_8/stack_1:output:01while/gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/gru_cell_1/MatMul_5MatMulwhile/gru_cell_1/mul:z:0)while/gru_cell_1/strided_slice_8:output:0*
T0*
_output_shapes
:	
while/gru_cell_1/add_2AddV2#while/gru_cell_1/BiasAdd_2:output:0#while/gru_cell_1/MatMul_5:product:0*
T0*
_output_shapes
:	c
while/gru_cell_1/TanhTanhwhile/gru_cell_1/add_2:z:0*
T0*
_output_shapes
:	z
while/gru_cell_1/mul_1Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*
_output_shapes
:	[
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*
_output_shapes
:	|
while/gru_cell_1/mul_2Mulwhile/gru_cell_1/sub:z:0while/gru_cell_1/Tanh:y:0*
T0*
_output_shapes
:	
while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_1:z:0while/gru_cell_1/mul_2:z:0*
T0*
_output_shapes
:	Ă
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:éčŇM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éčŇo
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0^while/NoOp*
T0*
_output_shapes
:	

while/NoOpNoOp ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6"^while/gru_cell_1/ReadVariableOp_7"^while/gru_cell_1/ReadVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 "Z
*while_gru_cell_1_readvariableop_3_resource,while_gru_cell_1_readvariableop_3_resource_0"Z
*while_gru_cell_1_readvariableop_6_resource,while_gru_cell_1_readvariableop_6_resource_0"V
(while_gru_cell_1_readvariableop_resource*while_gru_cell_1_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*0
_input_shapes
: : : : :	: : : : : 2B
while/gru_cell_1/ReadVariableOpwhile/gru_cell_1/ReadVariableOp2F
!while/gru_cell_1/ReadVariableOp_1!while/gru_cell_1/ReadVariableOp_12F
!while/gru_cell_1/ReadVariableOp_2!while/gru_cell_1/ReadVariableOp_22F
!while/gru_cell_1/ReadVariableOp_3!while/gru_cell_1/ReadVariableOp_32F
!while/gru_cell_1/ReadVariableOp_4!while/gru_cell_1/ReadVariableOp_42F
!while/gru_cell_1/ReadVariableOp_5!while/gru_cell_1/ReadVariableOp_52F
!while/gru_cell_1/ReadVariableOp_6!while/gru_cell_1/ReadVariableOp_62F
!while/gru_cell_1/ReadVariableOp_7!while/gru_cell_1/ReadVariableOp_72F
!while/gru_cell_1/ReadVariableOp_8!while/gru_cell_1/ReadVariableOp_8: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: 
çv
Ť	
while_body_290349
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
*while_gru_cell_1_readvariableop_resource_0:
;
,while_gru_cell_1_readvariableop_3_resource_0:	@
,while_gru_cell_1_readvariableop_6_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
(while_gru_cell_1_readvariableop_resource:
9
*while_gru_cell_1_readvariableop_3_resource:	>
*while_gru_cell_1_readvariableop_6_resource:
˘while/gru_cell_1/ReadVariableOp˘!while/gru_cell_1/ReadVariableOp_1˘!while/gru_cell_1/ReadVariableOp_2˘!while/gru_cell_1/ReadVariableOp_3˘!while/gru_cell_1/ReadVariableOp_4˘!while/gru_cell_1/ReadVariableOp_5˘!while/gru_cell_1/ReadVariableOp_6˘!while/gru_cell_1/ReadVariableOp_7˘!while/gru_cell_1/ReadVariableOp_8
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	*
element_dtype0
while/gru_cell_1/ReadVariableOpReadVariableOp*while_gru_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0u
$while/gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        w
&while/gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       w
&while/gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Â
while/gru_cell_1/strided_sliceStridedSlice'while/gru_cell_1/ReadVariableOp:value:0-while/gru_cell_1/strided_slice/stack:output:0/while/gru_cell_1/strided_slice/stack_1:output:0/while/gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskŚ
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell_1/strided_slice:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_1ReadVariableOp*while_gru_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0w
&while/gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/gru_cell_1/strided_slice_1StridedSlice)while/gru_cell_1/ReadVariableOp_1:value:0/while/gru_cell_1/strided_slice_1/stack:output:01while/gru_cell_1/strided_slice_1/stack_1:output:01while/gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskŞ
while/gru_cell_1/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_1/strided_slice_1:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_2ReadVariableOp*while_gru_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0w
&while/gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        y
(while/gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/gru_cell_1/strided_slice_2StridedSlice)while/gru_cell_1/ReadVariableOp_2:value:0/while/gru_cell_1/strided_slice_2/stack:output:01while/gru_cell_1/strided_slice_2/stack_1:output:01while/gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskŞ
while/gru_cell_1/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_1/strided_slice_2:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_3ReadVariableOp,while_gru_cell_1_readvariableop_3_resource_0*
_output_shapes	
:*
dtype0p
&while/gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: s
(while/gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(while/gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ˇ
 while/gru_cell_1/strided_slice_3StridedSlice)while/gru_cell_1/ReadVariableOp_3:value:0/while/gru_cell_1/strided_slice_3/stack:output:01while/gru_cell_1/strided_slice_3/stack_1:output:01while/gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*

begin_mask
while/gru_cell_1/BiasAddBiasAdd!while/gru_cell_1/MatMul:product:0)while/gru_cell_1/strided_slice_3:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_4ReadVariableOp,while_gru_cell_1_readvariableop_3_resource_0*
_output_shapes	
:*
dtype0q
&while/gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:s
(while/gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(while/gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ľ
 while/gru_cell_1/strided_slice_4StridedSlice)while/gru_cell_1/ReadVariableOp_4:value:0/while/gru_cell_1/strided_slice_4/stack:output:01while/gru_cell_1/strided_slice_4/stack_1:output:01while/gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:
while/gru_cell_1/BiasAdd_1BiasAdd#while/gru_cell_1/MatMul_1:product:0)while/gru_cell_1/strided_slice_4:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_5ReadVariableOp,while_gru_cell_1_readvariableop_3_resource_0*
_output_shapes	
:*
dtype0q
&while/gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(while/gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(while/gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ľ
 while/gru_cell_1/strided_slice_5StridedSlice)while/gru_cell_1/ReadVariableOp_5:value:0/while/gru_cell_1/strided_slice_5/stack:output:01while/gru_cell_1/strided_slice_5/stack_1:output:01while/gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*
end_mask
while/gru_cell_1/BiasAdd_2BiasAdd#while/gru_cell_1/MatMul_2:product:0)while/gru_cell_1/strided_slice_5:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_6ReadVariableOp,while_gru_cell_1_readvariableop_6_resource_0* 
_output_shapes
:
*
dtype0w
&while/gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/gru_cell_1/strided_slice_6StridedSlice)while/gru_cell_1/ReadVariableOp_6:value:0/while/gru_cell_1/strided_slice_6/stack:output:01while/gru_cell_1/strided_slice_6/stack_1:output:01while/gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/gru_cell_1/MatMul_3MatMulwhile_placeholder_2)while/gru_cell_1/strided_slice_6:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_7ReadVariableOp,while_gru_cell_1_readvariableop_6_resource_0* 
_output_shapes
:
*
dtype0w
&while/gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/gru_cell_1/strided_slice_7StridedSlice)while/gru_cell_1/ReadVariableOp_7:value:0/while/gru_cell_1/strided_slice_7/stack:output:01while/gru_cell_1/strided_slice_7/stack_1:output:01while/gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/gru_cell_1/MatMul_4MatMulwhile_placeholder_2)while/gru_cell_1/strided_slice_7:output:0*
T0*
_output_shapes
:	
while/gru_cell_1/addAddV2!while/gru_cell_1/BiasAdd:output:0#while/gru_cell_1/MatMul_3:product:0*
T0*
_output_shapes
:	g
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*
_output_shapes
:	
while/gru_cell_1/add_1AddV2#while/gru_cell_1/BiasAdd_1:output:0#while/gru_cell_1/MatMul_4:product:0*
T0*
_output_shapes
:	k
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*
_output_shapes
:	z
while/gru_cell_1/mulMulwhile/gru_cell_1/Sigmoid_1:y:0while_placeholder_2*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_8ReadVariableOp,while_gru_cell_1_readvariableop_6_resource_0* 
_output_shapes
:
*
dtype0w
&while/gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        y
(while/gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/gru_cell_1/strided_slice_8StridedSlice)while/gru_cell_1/ReadVariableOp_8:value:0/while/gru_cell_1/strided_slice_8/stack:output:01while/gru_cell_1/strided_slice_8/stack_1:output:01while/gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/gru_cell_1/MatMul_5MatMulwhile/gru_cell_1/mul:z:0)while/gru_cell_1/strided_slice_8:output:0*
T0*
_output_shapes
:	
while/gru_cell_1/add_2AddV2#while/gru_cell_1/BiasAdd_2:output:0#while/gru_cell_1/MatMul_5:product:0*
T0*
_output_shapes
:	c
while/gru_cell_1/TanhTanhwhile/gru_cell_1/add_2:z:0*
T0*
_output_shapes
:	z
while/gru_cell_1/mul_1Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*
_output_shapes
:	[
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*
_output_shapes
:	|
while/gru_cell_1/mul_2Mulwhile/gru_cell_1/sub:z:0while/gru_cell_1/Tanh:y:0*
T0*
_output_shapes
:	
while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_1:z:0while/gru_cell_1/mul_2:z:0*
T0*
_output_shapes
:	Ă
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:éčŇM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éčŇo
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0^while/NoOp*
T0*
_output_shapes
:	

while/NoOpNoOp ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6"^while/gru_cell_1/ReadVariableOp_7"^while/gru_cell_1/ReadVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 "Z
*while_gru_cell_1_readvariableop_3_resource,while_gru_cell_1_readvariableop_3_resource_0"Z
*while_gru_cell_1_readvariableop_6_resource,while_gru_cell_1_readvariableop_6_resource_0"V
(while_gru_cell_1_readvariableop_resource*while_gru_cell_1_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*0
_input_shapes
: : : : :	: : : : : 2B
while/gru_cell_1/ReadVariableOpwhile/gru_cell_1/ReadVariableOp2F
!while/gru_cell_1/ReadVariableOp_1!while/gru_cell_1/ReadVariableOp_12F
!while/gru_cell_1/ReadVariableOp_2!while/gru_cell_1/ReadVariableOp_22F
!while/gru_cell_1/ReadVariableOp_3!while/gru_cell_1/ReadVariableOp_32F
!while/gru_cell_1/ReadVariableOp_4!while/gru_cell_1/ReadVariableOp_42F
!while/gru_cell_1/ReadVariableOp_5!while/gru_cell_1/ReadVariableOp_52F
!while/gru_cell_1/ReadVariableOp_6!while/gru_cell_1/ReadVariableOp_62F
!while/gru_cell_1/ReadVariableOp_7!while/gru_cell_1/ReadVariableOp_72F
!while/gru_cell_1/ReadVariableOp_8!while/gru_cell_1/ReadVariableOp_8: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: 
Ć
¨
while_cond_289845
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice4
0while_while_cond_289845___redundant_placeholder04
0while_while_cond_289845___redundant_placeholder14
0while_while_cond_289845___redundant_placeholder24
0while_while_cond_289845___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :	: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
:
Ć
¨
while_cond_290037
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice4
0while_while_cond_290037___redundant_placeholder04
0while_while_cond_290037___redundant_placeholder14
0while_while_cond_290037___redundant_placeholder24
0while_while_cond_290037___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :	: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
:
ćN

F__inference_gru_cell_1_layer_call_and_return_conditional_losses_289799

inputs
states:	+
readvariableop_resource:
(
readvariableop_3_resource:	-
readvariableop_6_resource:

identity

identity_1˘MatMul_3/ReadVariableOp˘MatMul_4/ReadVariableOp˘ReadVariableOp˘ReadVariableOp_1˘ReadVariableOp_2˘ReadVariableOp_3˘ReadVariableOp_4˘ReadVariableOp_5˘ReadVariableOp_6˘ReadVariableOp_7˘ReadVariableOp_8˘mul/ReadVariableOp˘mul_1/ReadVariableOph
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      í
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskZ
MatMulMatMulinputsstrided_slice:output:0*
T0*
_output_shapes
:	j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask^
MatMul_1MatMulinputsstrided_slice_1:output:0*
T0*
_output_shapes
:	j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask^
MatMul_2MatMulinputsstrided_slice_2:output:0*
T0*
_output_shapes
:	g
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes	
:*
dtype0_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: b
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*

begin_maskh
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*
_output_shapes
:	g
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes	
:*
dtype0`
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Đ
strided_slice_4StridedSliceReadVariableOp_4:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:l
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*
_output_shapes
:	g
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes	
:*
dtype0`
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ŕ
strided_slice_5StridedSliceReadVariableOp_5:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*
end_maskl
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*
_output_shapes
:	l
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource* 
_output_shapes
:
*
dtype0f
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_6StridedSliceReadVariableOp_6:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask_
MatMul_3/ReadVariableOpReadVariableOpstates*
_output_shapes
:	*
dtype0w
MatMul_3MatMulMatMul_3/ReadVariableOp:value:0strided_slice_6:output:0*
T0*
_output_shapes
:	l
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource* 
_output_shapes
:
*
dtype0f
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_7StridedSliceReadVariableOp_7:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask_
MatMul_4/ReadVariableOpReadVariableOpstates*
_output_shapes
:	*
dtype0w
MatMul_4MatMulMatMul_4/ReadVariableOp:value:0strided_slice_7:output:0*
T0*
_output_shapes
:	\
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*
_output_shapes
:	E
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:	`
add_1AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*
_output_shapes
:	I
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:	Z
mul/ReadVariableOpReadVariableOpstates*
_output_shapes
:	*
dtype0_
mulMulSigmoid_1:y:0mul/ReadVariableOp:value:0*
T0*
_output_shapes
:	l
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource* 
_output_shapes
:
*
dtype0f
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_8StridedSliceReadVariableOp_8:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask_
MatMul_5MatMulmul:z:0strided_slice_8:output:0*
T0*
_output_shapes
:	`
add_2AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*
_output_shapes
:	A
TanhTanh	add_2:z:0*
T0*
_output_shapes
:	\
mul_1/ReadVariableOpReadVariableOpstates*
_output_shapes
:	*
dtype0a
mul_1MulSigmoid:y:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Q
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:	I
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:	N
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:	P
IdentityIdentity	add_3:z:0^NoOp*
T0*
_output_shapes
:	R

Identity_1Identity	add_3:z:0^NoOp*
T0*
_output_shapes
:	Ď
NoOpNoOp^MatMul_3/ReadVariableOp^MatMul_4/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8^mul/ReadVariableOp^mul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:	: : : : 22
MatMul_3/ReadVariableOpMatMul_3/ReadVariableOp22
MatMul_4/ReadVariableOpMatMul_4/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82(
mul/ReadVariableOpmul/ReadVariableOp2,
mul_1/ReadVariableOpmul_1/ReadVariableOp:G C

_output_shapes
:	
 
_user_specified_nameinputs:&"
 
_user_specified_namestates
Ć

gru_1_while_body_291008(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2%
!gru_1_while_gru_1_strided_slice_0c
_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0D
0gru_1_while_gru_cell_1_readvariableop_resource_0:
A
2gru_1_while_gru_cell_1_readvariableop_3_resource_0:	F
2gru_1_while_gru_cell_1_readvariableop_6_resource_0:

gru_1_while_identity
gru_1_while_identity_1
gru_1_while_identity_2
gru_1_while_identity_3
gru_1_while_identity_4#
gru_1_while_gru_1_strided_slicea
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensorB
.gru_1_while_gru_cell_1_readvariableop_resource:
?
0gru_1_while_gru_cell_1_readvariableop_3_resource:	D
0gru_1_while_gru_cell_1_readvariableop_6_resource:
˘%gru_1/while/gru_cell_1/ReadVariableOp˘'gru_1/while/gru_cell_1/ReadVariableOp_1˘'gru_1/while/gru_cell_1/ReadVariableOp_2˘'gru_1/while/gru_cell_1/ReadVariableOp_3˘'gru_1/while/gru_cell_1/ReadVariableOp_4˘'gru_1/while/gru_cell_1/ReadVariableOp_5˘'gru_1/while/gru_cell_1/ReadVariableOp_6˘'gru_1/while/gru_cell_1/ReadVariableOp_7˘'gru_1/while/gru_cell_1/ReadVariableOp_8
=gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ź
/gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0gru_1_while_placeholderFgru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	*
element_dtype0
%gru_1/while/gru_cell_1/ReadVariableOpReadVariableOp0gru_1_while_gru_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0{
*gru_1/while/gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        }
,gru_1/while/gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,gru_1/while/gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ŕ
$gru_1/while/gru_cell_1/strided_sliceStridedSlice-gru_1/while/gru_cell_1/ReadVariableOp:value:03gru_1/while/gru_cell_1/strided_slice/stack:output:05gru_1/while/gru_cell_1/strided_slice/stack_1:output:05gru_1/while/gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask¸
gru_1/while/gru_cell_1/MatMulMatMul6gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0-gru_1/while/gru_cell_1/strided_slice:output:0*
T0*
_output_shapes
:	
'gru_1/while/gru_cell_1/ReadVariableOp_1ReadVariableOp0gru_1_while_gru_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0}
,gru_1/while/gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
.gru_1/while/gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.gru_1/while/gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ę
&gru_1/while/gru_cell_1/strided_slice_1StridedSlice/gru_1/while/gru_cell_1/ReadVariableOp_1:value:05gru_1/while/gru_cell_1/strided_slice_1/stack:output:07gru_1/while/gru_cell_1/strided_slice_1/stack_1:output:07gru_1/while/gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskź
gru_1/while/gru_cell_1/MatMul_1MatMul6gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0/gru_1/while/gru_cell_1/strided_slice_1:output:0*
T0*
_output_shapes
:	
'gru_1/while/gru_cell_1/ReadVariableOp_2ReadVariableOp0gru_1_while_gru_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0}
,gru_1/while/gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
.gru_1/while/gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
.gru_1/while/gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ę
&gru_1/while/gru_cell_1/strided_slice_2StridedSlice/gru_1/while/gru_cell_1/ReadVariableOp_2:value:05gru_1/while/gru_cell_1/strided_slice_2/stack:output:07gru_1/while/gru_cell_1/strided_slice_2/stack_1:output:07gru_1/while/gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskź
gru_1/while/gru_cell_1/MatMul_2MatMul6gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0/gru_1/while/gru_cell_1/strided_slice_2:output:0*
T0*
_output_shapes
:	
'gru_1/while/gru_cell_1/ReadVariableOp_3ReadVariableOp2gru_1_while_gru_cell_1_readvariableop_3_resource_0*
_output_shapes	
:*
dtype0v
,gru_1/while/gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: y
.gru_1/while/gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.gru_1/while/gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ő
&gru_1/while/gru_cell_1/strided_slice_3StridedSlice/gru_1/while/gru_cell_1/ReadVariableOp_3:value:05gru_1/while/gru_cell_1/strided_slice_3/stack:output:07gru_1/while/gru_cell_1/strided_slice_3/stack_1:output:07gru_1/while/gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*

begin_mask­
gru_1/while/gru_cell_1/BiasAddBiasAdd'gru_1/while/gru_cell_1/MatMul:product:0/gru_1/while/gru_cell_1/strided_slice_3:output:0*
T0*
_output_shapes
:	
'gru_1/while/gru_cell_1/ReadVariableOp_4ReadVariableOp2gru_1_while_gru_cell_1_readvariableop_3_resource_0*
_output_shapes	
:*
dtype0w
,gru_1/while/gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:y
.gru_1/while/gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.gru_1/while/gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ă
&gru_1/while/gru_cell_1/strided_slice_4StridedSlice/gru_1/while/gru_cell_1/ReadVariableOp_4:value:05gru_1/while/gru_cell_1/strided_slice_4/stack:output:07gru_1/while/gru_cell_1/strided_slice_4/stack_1:output:07gru_1/while/gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:ą
 gru_1/while/gru_cell_1/BiasAdd_1BiasAdd)gru_1/while/gru_cell_1/MatMul_1:product:0/gru_1/while/gru_cell_1/strided_slice_4:output:0*
T0*
_output_shapes
:	
'gru_1/while/gru_cell_1/ReadVariableOp_5ReadVariableOp2gru_1_while_gru_cell_1_readvariableop_3_resource_0*
_output_shapes	
:*
dtype0w
,gru_1/while/gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.gru_1/while/gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.gru_1/while/gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ó
&gru_1/while/gru_cell_1/strided_slice_5StridedSlice/gru_1/while/gru_cell_1/ReadVariableOp_5:value:05gru_1/while/gru_cell_1/strided_slice_5/stack:output:07gru_1/while/gru_cell_1/strided_slice_5/stack_1:output:07gru_1/while/gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*
end_maską
 gru_1/while/gru_cell_1/BiasAdd_2BiasAdd)gru_1/while/gru_cell_1/MatMul_2:product:0/gru_1/while/gru_cell_1/strided_slice_5:output:0*
T0*
_output_shapes
:	
'gru_1/while/gru_cell_1/ReadVariableOp_6ReadVariableOp2gru_1_while_gru_cell_1_readvariableop_6_resource_0* 
_output_shapes
:
*
dtype0}
,gru_1/while/gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.gru_1/while/gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.gru_1/while/gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ę
&gru_1/while/gru_cell_1/strided_slice_6StridedSlice/gru_1/while/gru_cell_1/ReadVariableOp_6:value:05gru_1/while/gru_cell_1/strided_slice_6/stack:output:07gru_1/while/gru_cell_1/strided_slice_6/stack_1:output:07gru_1/while/gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_1/while/gru_cell_1/MatMul_3MatMulgru_1_while_placeholder_2/gru_1/while/gru_cell_1/strided_slice_6:output:0*
T0*
_output_shapes
:	
'gru_1/while/gru_cell_1/ReadVariableOp_7ReadVariableOp2gru_1_while_gru_cell_1_readvariableop_6_resource_0* 
_output_shapes
:
*
dtype0}
,gru_1/while/gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       
.gru_1/while/gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.gru_1/while/gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ę
&gru_1/while/gru_cell_1/strided_slice_7StridedSlice/gru_1/while/gru_cell_1/ReadVariableOp_7:value:05gru_1/while/gru_cell_1/strided_slice_7/stack:output:07gru_1/while/gru_cell_1/strided_slice_7/stack_1:output:07gru_1/while/gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_1/while/gru_cell_1/MatMul_4MatMulgru_1_while_placeholder_2/gru_1/while/gru_cell_1/strided_slice_7:output:0*
T0*
_output_shapes
:	Ą
gru_1/while/gru_cell_1/addAddV2'gru_1/while/gru_cell_1/BiasAdd:output:0)gru_1/while/gru_cell_1/MatMul_3:product:0*
T0*
_output_shapes
:	s
gru_1/while/gru_cell_1/SigmoidSigmoidgru_1/while/gru_cell_1/add:z:0*
T0*
_output_shapes
:	Ľ
gru_1/while/gru_cell_1/add_1AddV2)gru_1/while/gru_cell_1/BiasAdd_1:output:0)gru_1/while/gru_cell_1/MatMul_4:product:0*
T0*
_output_shapes
:	w
 gru_1/while/gru_cell_1/Sigmoid_1Sigmoid gru_1/while/gru_cell_1/add_1:z:0*
T0*
_output_shapes
:	
gru_1/while/gru_cell_1/mulMul$gru_1/while/gru_cell_1/Sigmoid_1:y:0gru_1_while_placeholder_2*
T0*
_output_shapes
:	
'gru_1/while/gru_cell_1/ReadVariableOp_8ReadVariableOp2gru_1_while_gru_cell_1_readvariableop_6_resource_0* 
_output_shapes
:
*
dtype0}
,gru_1/while/gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       
.gru_1/while/gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
.gru_1/while/gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ę
&gru_1/while/gru_cell_1/strided_slice_8StridedSlice/gru_1/while/gru_cell_1/ReadVariableOp_8:value:05gru_1/while/gru_cell_1/strided_slice_8/stack:output:07gru_1/while/gru_cell_1/strided_slice_8/stack_1:output:07gru_1/while/gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask¤
gru_1/while/gru_cell_1/MatMul_5MatMulgru_1/while/gru_cell_1/mul:z:0/gru_1/while/gru_cell_1/strided_slice_8:output:0*
T0*
_output_shapes
:	Ľ
gru_1/while/gru_cell_1/add_2AddV2)gru_1/while/gru_cell_1/BiasAdd_2:output:0)gru_1/while/gru_cell_1/MatMul_5:product:0*
T0*
_output_shapes
:	o
gru_1/while/gru_cell_1/TanhTanh gru_1/while/gru_cell_1/add_2:z:0*
T0*
_output_shapes
:	
gru_1/while/gru_cell_1/mul_1Mul"gru_1/while/gru_cell_1/Sigmoid:y:0gru_1_while_placeholder_2*
T0*
_output_shapes
:	a
gru_1/while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_1/while/gru_cell_1/subSub%gru_1/while/gru_cell_1/sub/x:output:0"gru_1/while/gru_cell_1/Sigmoid:y:0*
T0*
_output_shapes
:	
gru_1/while/gru_cell_1/mul_2Mulgru_1/while/gru_cell_1/sub:z:0gru_1/while/gru_cell_1/Tanh:y:0*
T0*
_output_shapes
:	
gru_1/while/gru_cell_1/add_3AddV2 gru_1/while/gru_cell_1/mul_1:z:0 gru_1/while/gru_cell_1/mul_2:z:0*
T0*
_output_shapes
:	Ű
0gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_1_while_placeholder_1gru_1_while_placeholder gru_1/while/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:éčŇS
gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :n
gru_1/while/addAddV2gru_1_while_placeholdergru_1/while/add/y:output:0*
T0*
_output_shapes
: U
gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
gru_1/while/add_1AddV2$gru_1_while_gru_1_while_loop_countergru_1/while/add_1/y:output:0*
T0*
_output_shapes
: k
gru_1/while/IdentityIdentitygru_1/while/add_1:z:0^gru_1/while/NoOp*
T0*
_output_shapes
: 
gru_1/while/Identity_1Identity*gru_1_while_gru_1_while_maximum_iterations^gru_1/while/NoOp*
T0*
_output_shapes
: k
gru_1/while/Identity_2Identitygru_1/while/add:z:0^gru_1/while/NoOp*
T0*
_output_shapes
: Ť
gru_1/while/Identity_3Identity@gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_1/while/NoOp*
T0*
_output_shapes
: :éčŇ
gru_1/while/Identity_4Identity gru_1/while/gru_cell_1/add_3:z:0^gru_1/while/NoOp*
T0*
_output_shapes
:	Ę
gru_1/while/NoOpNoOp&^gru_1/while/gru_cell_1/ReadVariableOp(^gru_1/while/gru_cell_1/ReadVariableOp_1(^gru_1/while/gru_cell_1/ReadVariableOp_2(^gru_1/while/gru_cell_1/ReadVariableOp_3(^gru_1/while/gru_cell_1/ReadVariableOp_4(^gru_1/while/gru_cell_1/ReadVariableOp_5(^gru_1/while/gru_cell_1/ReadVariableOp_6(^gru_1/while/gru_cell_1/ReadVariableOp_7(^gru_1/while/gru_cell_1/ReadVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 "D
gru_1_while_gru_1_strided_slice!gru_1_while_gru_1_strided_slice_0"f
0gru_1_while_gru_cell_1_readvariableop_3_resource2gru_1_while_gru_cell_1_readvariableop_3_resource_0"f
0gru_1_while_gru_cell_1_readvariableop_6_resource2gru_1_while_gru_cell_1_readvariableop_6_resource_0"b
.gru_1_while_gru_cell_1_readvariableop_resource0gru_1_while_gru_cell_1_readvariableop_resource_0"5
gru_1_while_identitygru_1/while/Identity:output:0"9
gru_1_while_identity_1gru_1/while/Identity_1:output:0"9
gru_1_while_identity_2gru_1/while/Identity_2:output:0"9
gru_1_while_identity_3gru_1/while/Identity_3:output:0"9
gru_1_while_identity_4gru_1/while/Identity_4:output:0"Ŕ
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*0
_input_shapes
: : : : :	: : : : : 2N
%gru_1/while/gru_cell_1/ReadVariableOp%gru_1/while/gru_cell_1/ReadVariableOp2R
'gru_1/while/gru_cell_1/ReadVariableOp_1'gru_1/while/gru_cell_1/ReadVariableOp_12R
'gru_1/while/gru_cell_1/ReadVariableOp_2'gru_1/while/gru_cell_1/ReadVariableOp_22R
'gru_1/while/gru_cell_1/ReadVariableOp_3'gru_1/while/gru_cell_1/ReadVariableOp_32R
'gru_1/while/gru_cell_1/ReadVariableOp_4'gru_1/while/gru_cell_1/ReadVariableOp_42R
'gru_1/while/gru_cell_1/ReadVariableOp_5'gru_1/while/gru_cell_1/ReadVariableOp_52R
'gru_1/while/gru_cell_1/ReadVariableOp_6'gru_1/while/gru_cell_1/ReadVariableOp_62R
'gru_1/while/gru_cell_1/ReadVariableOp_7'gru_1/while/gru_cell_1/ReadVariableOp_72R
'gru_1/while/gru_cell_1/ReadVariableOp_8'gru_1/while/gru_cell_1/ReadVariableOp_8: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: 
ć
Ä
H__inference_sequential_1_layer_call_and_return_conditional_losses_290592
embedding_1_input%
embedding_1_290574:	V 
gru_1_290577:

gru_1_290579:	 
gru_1_290581:

gru_1_290583:	!
dense_1_290586:	V
dense_1_290588:V
identity˘dense_1/StatefulPartitionedCall˘#embedding_1/StatefulPartitionedCall˘gru_1/StatefulPartitionedCallů
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallembedding_1_inputembedding_1_290574*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_embedding_1_layer_call_and_return_conditional_losses_289938ą
gru_1/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0gru_1_290577gru_1_290579gru_1_290581gru_1_290583*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_290163
dense_1/StatefulPartitionedCallStatefulPartitionedCall&gru_1/StatefulPartitionedCall:output:0dense_1_290586dense_1_290588*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙V*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_290203{
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙VŽ
NoOpNoOp ^dense_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:˙˙˙˙˙˙˙˙˙: : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:Z V
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameembedding_1_input
Ő	
ş
-__inference_sequential_1_layer_call_fn_290571
embedding_1_input
unknown:	V
	unknown_0:

	unknown_1:	
	unknown_2:

	unknown_3:	
	unknown_4:	V
	unknown_5:V
identity˘StatefulPartitionedCallŻ
StatefulPartitionedCallStatefulPartitionedCallembedding_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙V*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_290535s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙V`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:˙˙˙˙˙˙˙˙˙: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameembedding_1_input
Ć
¨
while_cond_290348
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice4
0while_while_cond_290348___redundant_placeholder04
0while_while_cond_290348___redundant_placeholder14
0while_while_cond_290348___redundant_placeholder24
0while_while_cond_290348___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :	: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
:
­
×
$sequential_1_gru_1_while_body_289180B
>sequential_1_gru_1_while_sequential_1_gru_1_while_loop_counterH
Dsequential_1_gru_1_while_sequential_1_gru_1_while_maximum_iterations(
$sequential_1_gru_1_while_placeholder*
&sequential_1_gru_1_while_placeholder_1*
&sequential_1_gru_1_while_placeholder_2?
;sequential_1_gru_1_while_sequential_1_gru_1_strided_slice_0}
ysequential_1_gru_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_gru_1_tensorarrayunstack_tensorlistfromtensor_0Q
=sequential_1_gru_1_while_gru_cell_1_readvariableop_resource_0:
N
?sequential_1_gru_1_while_gru_cell_1_readvariableop_3_resource_0:	S
?sequential_1_gru_1_while_gru_cell_1_readvariableop_6_resource_0:
%
!sequential_1_gru_1_while_identity'
#sequential_1_gru_1_while_identity_1'
#sequential_1_gru_1_while_identity_2'
#sequential_1_gru_1_while_identity_3'
#sequential_1_gru_1_while_identity_4=
9sequential_1_gru_1_while_sequential_1_gru_1_strided_slice{
wsequential_1_gru_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_gru_1_tensorarrayunstack_tensorlistfromtensorO
;sequential_1_gru_1_while_gru_cell_1_readvariableop_resource:
L
=sequential_1_gru_1_while_gru_cell_1_readvariableop_3_resource:	Q
=sequential_1_gru_1_while_gru_cell_1_readvariableop_6_resource:
˘2sequential_1/gru_1/while/gru_cell_1/ReadVariableOp˘4sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_1˘4sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_2˘4sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_3˘4sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_4˘4sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_5˘4sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_6˘4sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_7˘4sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_8
Jsequential_1/gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ý
<sequential_1/gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemysequential_1_gru_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_gru_1_tensorarrayunstack_tensorlistfromtensor_0$sequential_1_gru_1_while_placeholderSsequential_1/gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	*
element_dtype0˛
2sequential_1/gru_1/while/gru_cell_1/ReadVariableOpReadVariableOp=sequential_1_gru_1_while_gru_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
7sequential_1/gru_1/while/gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
9sequential_1/gru_1/while/gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
9sequential_1/gru_1/while/gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ą
1sequential_1/gru_1/while/gru_cell_1/strided_sliceStridedSlice:sequential_1/gru_1/while/gru_cell_1/ReadVariableOp:value:0@sequential_1/gru_1/while/gru_cell_1/strided_slice/stack:output:0Bsequential_1/gru_1/while/gru_cell_1/strided_slice/stack_1:output:0Bsequential_1/gru_1/while/gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskß
*sequential_1/gru_1/while/gru_cell_1/MatMulMatMulCsequential_1/gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0:sequential_1/gru_1/while/gru_cell_1/strided_slice:output:0*
T0*
_output_shapes
:	´
4sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_1ReadVariableOp=sequential_1_gru_1_while_gru_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
9sequential_1/gru_1/while/gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
;sequential_1/gru_1/while/gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
;sequential_1/gru_1/while/gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ť
3sequential_1/gru_1/while/gru_cell_1/strided_slice_1StridedSlice<sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_1:value:0Bsequential_1/gru_1/while/gru_cell_1/strided_slice_1/stack:output:0Dsequential_1/gru_1/while/gru_cell_1/strided_slice_1/stack_1:output:0Dsequential_1/gru_1/while/gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskă
,sequential_1/gru_1/while/gru_cell_1/MatMul_1MatMulCsequential_1/gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0<sequential_1/gru_1/while/gru_cell_1/strided_slice_1:output:0*
T0*
_output_shapes
:	´
4sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_2ReadVariableOp=sequential_1_gru_1_while_gru_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
9sequential_1/gru_1/while/gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
;sequential_1/gru_1/while/gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
;sequential_1/gru_1/while/gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ť
3sequential_1/gru_1/while/gru_cell_1/strided_slice_2StridedSlice<sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_2:value:0Bsequential_1/gru_1/while/gru_cell_1/strided_slice_2/stack:output:0Dsequential_1/gru_1/while/gru_cell_1/strided_slice_2/stack_1:output:0Dsequential_1/gru_1/while/gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskă
,sequential_1/gru_1/while/gru_cell_1/MatMul_2MatMulCsequential_1/gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0<sequential_1/gru_1/while/gru_cell_1/strided_slice_2:output:0*
T0*
_output_shapes
:	ą
4sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_3ReadVariableOp?sequential_1_gru_1_while_gru_cell_1_readvariableop_3_resource_0*
_output_shapes	
:*
dtype0
9sequential_1/gru_1/while/gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 
;sequential_1/gru_1/while/gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
;sequential_1/gru_1/while/gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3sequential_1/gru_1/while/gru_cell_1/strided_slice_3StridedSlice<sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_3:value:0Bsequential_1/gru_1/while/gru_cell_1/strided_slice_3/stack:output:0Dsequential_1/gru_1/while/gru_cell_1/strided_slice_3/stack_1:output:0Dsequential_1/gru_1/while/gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*

begin_maskÔ
+sequential_1/gru_1/while/gru_cell_1/BiasAddBiasAdd4sequential_1/gru_1/while/gru_cell_1/MatMul:product:0<sequential_1/gru_1/while/gru_cell_1/strided_slice_3:output:0*
T0*
_output_shapes
:	ą
4sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_4ReadVariableOp?sequential_1_gru_1_while_gru_cell_1_readvariableop_3_resource_0*
_output_shapes	
:*
dtype0
9sequential_1/gru_1/while/gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
;sequential_1/gru_1/while/gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
;sequential_1/gru_1/while/gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3sequential_1/gru_1/while/gru_cell_1/strided_slice_4StridedSlice<sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_4:value:0Bsequential_1/gru_1/while/gru_cell_1/strided_slice_4/stack:output:0Dsequential_1/gru_1/while/gru_cell_1/strided_slice_4/stack_1:output:0Dsequential_1/gru_1/while/gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:Ř
-sequential_1/gru_1/while/gru_cell_1/BiasAdd_1BiasAdd6sequential_1/gru_1/while/gru_cell_1/MatMul_1:product:0<sequential_1/gru_1/while/gru_cell_1/strided_slice_4:output:0*
T0*
_output_shapes
:	ą
4sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_5ReadVariableOp?sequential_1_gru_1_while_gru_cell_1_readvariableop_3_resource_0*
_output_shapes	
:*
dtype0
9sequential_1/gru_1/while/gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:
;sequential_1/gru_1/while/gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
;sequential_1/gru_1/while/gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3sequential_1/gru_1/while/gru_cell_1/strided_slice_5StridedSlice<sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_5:value:0Bsequential_1/gru_1/while/gru_cell_1/strided_slice_5/stack:output:0Dsequential_1/gru_1/while/gru_cell_1/strided_slice_5/stack_1:output:0Dsequential_1/gru_1/while/gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*
end_maskŘ
-sequential_1/gru_1/while/gru_cell_1/BiasAdd_2BiasAdd6sequential_1/gru_1/while/gru_cell_1/MatMul_2:product:0<sequential_1/gru_1/while/gru_cell_1/strided_slice_5:output:0*
T0*
_output_shapes
:	ś
4sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_6ReadVariableOp?sequential_1_gru_1_while_gru_cell_1_readvariableop_6_resource_0* 
_output_shapes
:
*
dtype0
9sequential_1/gru_1/while/gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        
;sequential_1/gru_1/while/gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
;sequential_1/gru_1/while/gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ť
3sequential_1/gru_1/while/gru_cell_1/strided_slice_6StridedSlice<sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_6:value:0Bsequential_1/gru_1/while/gru_cell_1/strided_slice_6/stack:output:0Dsequential_1/gru_1/while/gru_cell_1/strided_slice_6/stack_1:output:0Dsequential_1/gru_1/while/gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskĆ
,sequential_1/gru_1/while/gru_cell_1/MatMul_3MatMul&sequential_1_gru_1_while_placeholder_2<sequential_1/gru_1/while/gru_cell_1/strided_slice_6:output:0*
T0*
_output_shapes
:	ś
4sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_7ReadVariableOp?sequential_1_gru_1_while_gru_cell_1_readvariableop_6_resource_0* 
_output_shapes
:
*
dtype0
9sequential_1/gru_1/while/gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       
;sequential_1/gru_1/while/gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
;sequential_1/gru_1/while/gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ť
3sequential_1/gru_1/while/gru_cell_1/strided_slice_7StridedSlice<sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_7:value:0Bsequential_1/gru_1/while/gru_cell_1/strided_slice_7/stack:output:0Dsequential_1/gru_1/while/gru_cell_1/strided_slice_7/stack_1:output:0Dsequential_1/gru_1/while/gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskĆ
,sequential_1/gru_1/while/gru_cell_1/MatMul_4MatMul&sequential_1_gru_1_while_placeholder_2<sequential_1/gru_1/while/gru_cell_1/strided_slice_7:output:0*
T0*
_output_shapes
:	Č
'sequential_1/gru_1/while/gru_cell_1/addAddV24sequential_1/gru_1/while/gru_cell_1/BiasAdd:output:06sequential_1/gru_1/while/gru_cell_1/MatMul_3:product:0*
T0*
_output_shapes
:	
+sequential_1/gru_1/while/gru_cell_1/SigmoidSigmoid+sequential_1/gru_1/while/gru_cell_1/add:z:0*
T0*
_output_shapes
:	Ě
)sequential_1/gru_1/while/gru_cell_1/add_1AddV26sequential_1/gru_1/while/gru_cell_1/BiasAdd_1:output:06sequential_1/gru_1/while/gru_cell_1/MatMul_4:product:0*
T0*
_output_shapes
:	
-sequential_1/gru_1/while/gru_cell_1/Sigmoid_1Sigmoid-sequential_1/gru_1/while/gru_cell_1/add_1:z:0*
T0*
_output_shapes
:	ł
'sequential_1/gru_1/while/gru_cell_1/mulMul1sequential_1/gru_1/while/gru_cell_1/Sigmoid_1:y:0&sequential_1_gru_1_while_placeholder_2*
T0*
_output_shapes
:	ś
4sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_8ReadVariableOp?sequential_1_gru_1_while_gru_cell_1_readvariableop_6_resource_0* 
_output_shapes
:
*
dtype0
9sequential_1/gru_1/while/gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       
;sequential_1/gru_1/while/gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
;sequential_1/gru_1/while/gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ť
3sequential_1/gru_1/while/gru_cell_1/strided_slice_8StridedSlice<sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_8:value:0Bsequential_1/gru_1/while/gru_cell_1/strided_slice_8/stack:output:0Dsequential_1/gru_1/while/gru_cell_1/strided_slice_8/stack_1:output:0Dsequential_1/gru_1/while/gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskË
,sequential_1/gru_1/while/gru_cell_1/MatMul_5MatMul+sequential_1/gru_1/while/gru_cell_1/mul:z:0<sequential_1/gru_1/while/gru_cell_1/strided_slice_8:output:0*
T0*
_output_shapes
:	Ě
)sequential_1/gru_1/while/gru_cell_1/add_2AddV26sequential_1/gru_1/while/gru_cell_1/BiasAdd_2:output:06sequential_1/gru_1/while/gru_cell_1/MatMul_5:product:0*
T0*
_output_shapes
:	
(sequential_1/gru_1/while/gru_cell_1/TanhTanh-sequential_1/gru_1/while/gru_cell_1/add_2:z:0*
T0*
_output_shapes
:	ł
)sequential_1/gru_1/while/gru_cell_1/mul_1Mul/sequential_1/gru_1/while/gru_cell_1/Sigmoid:y:0&sequential_1_gru_1_while_placeholder_2*
T0*
_output_shapes
:	n
)sequential_1/gru_1/while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?˝
'sequential_1/gru_1/while/gru_cell_1/subSub2sequential_1/gru_1/while/gru_cell_1/sub/x:output:0/sequential_1/gru_1/while/gru_cell_1/Sigmoid:y:0*
T0*
_output_shapes
:	ľ
)sequential_1/gru_1/while/gru_cell_1/mul_2Mul+sequential_1/gru_1/while/gru_cell_1/sub:z:0,sequential_1/gru_1/while/gru_cell_1/Tanh:y:0*
T0*
_output_shapes
:	ş
)sequential_1/gru_1/while/gru_cell_1/add_3AddV2-sequential_1/gru_1/while/gru_cell_1/mul_1:z:0-sequential_1/gru_1/while/gru_cell_1/mul_2:z:0*
T0*
_output_shapes
:	
=sequential_1/gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem&sequential_1_gru_1_while_placeholder_1$sequential_1_gru_1_while_placeholder-sequential_1/gru_1/while/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:éčŇ`
sequential_1/gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
sequential_1/gru_1/while/addAddV2$sequential_1_gru_1_while_placeholder'sequential_1/gru_1/while/add/y:output:0*
T0*
_output_shapes
: b
 sequential_1/gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :ł
sequential_1/gru_1/while/add_1AddV2>sequential_1_gru_1_while_sequential_1_gru_1_while_loop_counter)sequential_1/gru_1/while/add_1/y:output:0*
T0*
_output_shapes
: 
!sequential_1/gru_1/while/IdentityIdentity"sequential_1/gru_1/while/add_1:z:0^sequential_1/gru_1/while/NoOp*
T0*
_output_shapes
: ś
#sequential_1/gru_1/while/Identity_1IdentityDsequential_1_gru_1_while_sequential_1_gru_1_while_maximum_iterations^sequential_1/gru_1/while/NoOp*
T0*
_output_shapes
: 
#sequential_1/gru_1/while/Identity_2Identity sequential_1/gru_1/while/add:z:0^sequential_1/gru_1/while/NoOp*
T0*
_output_shapes
: Ň
#sequential_1/gru_1/while/Identity_3IdentityMsequential_1/gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_1/gru_1/while/NoOp*
T0*
_output_shapes
: :éčŇ¨
#sequential_1/gru_1/while/Identity_4Identity-sequential_1/gru_1/while/gru_cell_1/add_3:z:0^sequential_1/gru_1/while/NoOp*
T0*
_output_shapes
:	Ě
sequential_1/gru_1/while/NoOpNoOp3^sequential_1/gru_1/while/gru_cell_1/ReadVariableOp5^sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_15^sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_25^sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_35^sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_45^sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_55^sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_65^sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_75^sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 "
=sequential_1_gru_1_while_gru_cell_1_readvariableop_3_resource?sequential_1_gru_1_while_gru_cell_1_readvariableop_3_resource_0"
=sequential_1_gru_1_while_gru_cell_1_readvariableop_6_resource?sequential_1_gru_1_while_gru_cell_1_readvariableop_6_resource_0"|
;sequential_1_gru_1_while_gru_cell_1_readvariableop_resource=sequential_1_gru_1_while_gru_cell_1_readvariableop_resource_0"O
!sequential_1_gru_1_while_identity*sequential_1/gru_1/while/Identity:output:0"S
#sequential_1_gru_1_while_identity_1,sequential_1/gru_1/while/Identity_1:output:0"S
#sequential_1_gru_1_while_identity_2,sequential_1/gru_1/while/Identity_2:output:0"S
#sequential_1_gru_1_while_identity_3,sequential_1/gru_1/while/Identity_3:output:0"S
#sequential_1_gru_1_while_identity_4,sequential_1/gru_1/while/Identity_4:output:0"x
9sequential_1_gru_1_while_sequential_1_gru_1_strided_slice;sequential_1_gru_1_while_sequential_1_gru_1_strided_slice_0"ô
wsequential_1_gru_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_gru_1_tensorarrayunstack_tensorlistfromtensorysequential_1_gru_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*0
_input_shapes
: : : : :	: : : : : 2h
2sequential_1/gru_1/while/gru_cell_1/ReadVariableOp2sequential_1/gru_1/while/gru_cell_1/ReadVariableOp2l
4sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_14sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_12l
4sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_24sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_22l
4sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_34sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_32l
4sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_44sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_42l
4sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_54sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_52l
4sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_64sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_62l
4sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_74sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_72l
4sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_84sequential_1/gru_1/while/gru_cell_1/ReadVariableOp_8: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: 
çv
Ť	
while_body_292012
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
*while_gru_cell_1_readvariableop_resource_0:
;
,while_gru_cell_1_readvariableop_3_resource_0:	@
,while_gru_cell_1_readvariableop_6_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
(while_gru_cell_1_readvariableop_resource:
9
*while_gru_cell_1_readvariableop_3_resource:	>
*while_gru_cell_1_readvariableop_6_resource:
˘while/gru_cell_1/ReadVariableOp˘!while/gru_cell_1/ReadVariableOp_1˘!while/gru_cell_1/ReadVariableOp_2˘!while/gru_cell_1/ReadVariableOp_3˘!while/gru_cell_1/ReadVariableOp_4˘!while/gru_cell_1/ReadVariableOp_5˘!while/gru_cell_1/ReadVariableOp_6˘!while/gru_cell_1/ReadVariableOp_7˘!while/gru_cell_1/ReadVariableOp_8
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	*
element_dtype0
while/gru_cell_1/ReadVariableOpReadVariableOp*while_gru_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0u
$while/gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        w
&while/gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       w
&while/gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Â
while/gru_cell_1/strided_sliceStridedSlice'while/gru_cell_1/ReadVariableOp:value:0-while/gru_cell_1/strided_slice/stack:output:0/while/gru_cell_1/strided_slice/stack_1:output:0/while/gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskŚ
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell_1/strided_slice:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_1ReadVariableOp*while_gru_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0w
&while/gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/gru_cell_1/strided_slice_1StridedSlice)while/gru_cell_1/ReadVariableOp_1:value:0/while/gru_cell_1/strided_slice_1/stack:output:01while/gru_cell_1/strided_slice_1/stack_1:output:01while/gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskŞ
while/gru_cell_1/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_1/strided_slice_1:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_2ReadVariableOp*while_gru_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0w
&while/gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        y
(while/gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/gru_cell_1/strided_slice_2StridedSlice)while/gru_cell_1/ReadVariableOp_2:value:0/while/gru_cell_1/strided_slice_2/stack:output:01while/gru_cell_1/strided_slice_2/stack_1:output:01while/gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskŞ
while/gru_cell_1/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_1/strided_slice_2:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_3ReadVariableOp,while_gru_cell_1_readvariableop_3_resource_0*
_output_shapes	
:*
dtype0p
&while/gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: s
(while/gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(while/gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ˇ
 while/gru_cell_1/strided_slice_3StridedSlice)while/gru_cell_1/ReadVariableOp_3:value:0/while/gru_cell_1/strided_slice_3/stack:output:01while/gru_cell_1/strided_slice_3/stack_1:output:01while/gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*

begin_mask
while/gru_cell_1/BiasAddBiasAdd!while/gru_cell_1/MatMul:product:0)while/gru_cell_1/strided_slice_3:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_4ReadVariableOp,while_gru_cell_1_readvariableop_3_resource_0*
_output_shapes	
:*
dtype0q
&while/gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:s
(while/gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(while/gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ľ
 while/gru_cell_1/strided_slice_4StridedSlice)while/gru_cell_1/ReadVariableOp_4:value:0/while/gru_cell_1/strided_slice_4/stack:output:01while/gru_cell_1/strided_slice_4/stack_1:output:01while/gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:
while/gru_cell_1/BiasAdd_1BiasAdd#while/gru_cell_1/MatMul_1:product:0)while/gru_cell_1/strided_slice_4:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_5ReadVariableOp,while_gru_cell_1_readvariableop_3_resource_0*
_output_shapes	
:*
dtype0q
&while/gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(while/gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(while/gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ľ
 while/gru_cell_1/strided_slice_5StridedSlice)while/gru_cell_1/ReadVariableOp_5:value:0/while/gru_cell_1/strided_slice_5/stack:output:01while/gru_cell_1/strided_slice_5/stack_1:output:01while/gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*
end_mask
while/gru_cell_1/BiasAdd_2BiasAdd#while/gru_cell_1/MatMul_2:product:0)while/gru_cell_1/strided_slice_5:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_6ReadVariableOp,while_gru_cell_1_readvariableop_6_resource_0* 
_output_shapes
:
*
dtype0w
&while/gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/gru_cell_1/strided_slice_6StridedSlice)while/gru_cell_1/ReadVariableOp_6:value:0/while/gru_cell_1/strided_slice_6/stack:output:01while/gru_cell_1/strided_slice_6/stack_1:output:01while/gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/gru_cell_1/MatMul_3MatMulwhile_placeholder_2)while/gru_cell_1/strided_slice_6:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_7ReadVariableOp,while_gru_cell_1_readvariableop_6_resource_0* 
_output_shapes
:
*
dtype0w
&while/gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/gru_cell_1/strided_slice_7StridedSlice)while/gru_cell_1/ReadVariableOp_7:value:0/while/gru_cell_1/strided_slice_7/stack:output:01while/gru_cell_1/strided_slice_7/stack_1:output:01while/gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/gru_cell_1/MatMul_4MatMulwhile_placeholder_2)while/gru_cell_1/strided_slice_7:output:0*
T0*
_output_shapes
:	
while/gru_cell_1/addAddV2!while/gru_cell_1/BiasAdd:output:0#while/gru_cell_1/MatMul_3:product:0*
T0*
_output_shapes
:	g
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*
_output_shapes
:	
while/gru_cell_1/add_1AddV2#while/gru_cell_1/BiasAdd_1:output:0#while/gru_cell_1/MatMul_4:product:0*
T0*
_output_shapes
:	k
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*
_output_shapes
:	z
while/gru_cell_1/mulMulwhile/gru_cell_1/Sigmoid_1:y:0while_placeholder_2*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_8ReadVariableOp,while_gru_cell_1_readvariableop_6_resource_0* 
_output_shapes
:
*
dtype0w
&while/gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        y
(while/gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/gru_cell_1/strided_slice_8StridedSlice)while/gru_cell_1/ReadVariableOp_8:value:0/while/gru_cell_1/strided_slice_8/stack:output:01while/gru_cell_1/strided_slice_8/stack_1:output:01while/gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/gru_cell_1/MatMul_5MatMulwhile/gru_cell_1/mul:z:0)while/gru_cell_1/strided_slice_8:output:0*
T0*
_output_shapes
:	
while/gru_cell_1/add_2AddV2#while/gru_cell_1/BiasAdd_2:output:0#while/gru_cell_1/MatMul_5:product:0*
T0*
_output_shapes
:	c
while/gru_cell_1/TanhTanhwhile/gru_cell_1/add_2:z:0*
T0*
_output_shapes
:	z
while/gru_cell_1/mul_1Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*
_output_shapes
:	[
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*
_output_shapes
:	|
while/gru_cell_1/mul_2Mulwhile/gru_cell_1/sub:z:0while/gru_cell_1/Tanh:y:0*
T0*
_output_shapes
:	
while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_1:z:0while/gru_cell_1/mul_2:z:0*
T0*
_output_shapes
:	Ă
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:éčŇM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éčŇo
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0^while/NoOp*
T0*
_output_shapes
:	

while/NoOpNoOp ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6"^while/gru_cell_1/ReadVariableOp_7"^while/gru_cell_1/ReadVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 "Z
*while_gru_cell_1_readvariableop_3_resource,while_gru_cell_1_readvariableop_3_resource_0"Z
*while_gru_cell_1_readvariableop_6_resource,while_gru_cell_1_readvariableop_6_resource_0"V
(while_gru_cell_1_readvariableop_resource*while_gru_cell_1_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*0
_input_shapes
: : : : :	: : : : : 2B
while/gru_cell_1/ReadVariableOpwhile/gru_cell_1/ReadVariableOp2F
!while/gru_cell_1/ReadVariableOp_1!while/gru_cell_1/ReadVariableOp_12F
!while/gru_cell_1/ReadVariableOp_2!while/gru_cell_1/ReadVariableOp_22F
!while/gru_cell_1/ReadVariableOp_3!while/gru_cell_1/ReadVariableOp_32F
!while/gru_cell_1/ReadVariableOp_4!while/gru_cell_1/ReadVariableOp_42F
!while/gru_cell_1/ReadVariableOp_5!while/gru_cell_1/ReadVariableOp_52F
!while/gru_cell_1/ReadVariableOp_6!while/gru_cell_1/ReadVariableOp_62F
!while/gru_cell_1/ReadVariableOp_7!while/gru_cell_1/ReadVariableOp_72F
!while/gru_cell_1/ReadVariableOp_8!while/gru_cell_1/ReadVariableOp_8: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: 
çv
Ť	
while_body_291346
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
*while_gru_cell_1_readvariableop_resource_0:
;
,while_gru_cell_1_readvariableop_3_resource_0:	@
,while_gru_cell_1_readvariableop_6_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
(while_gru_cell_1_readvariableop_resource:
9
*while_gru_cell_1_readvariableop_3_resource:	>
*while_gru_cell_1_readvariableop_6_resource:
˘while/gru_cell_1/ReadVariableOp˘!while/gru_cell_1/ReadVariableOp_1˘!while/gru_cell_1/ReadVariableOp_2˘!while/gru_cell_1/ReadVariableOp_3˘!while/gru_cell_1/ReadVariableOp_4˘!while/gru_cell_1/ReadVariableOp_5˘!while/gru_cell_1/ReadVariableOp_6˘!while/gru_cell_1/ReadVariableOp_7˘!while/gru_cell_1/ReadVariableOp_8
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	*
element_dtype0
while/gru_cell_1/ReadVariableOpReadVariableOp*while_gru_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0u
$while/gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        w
&while/gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       w
&while/gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Â
while/gru_cell_1/strided_sliceStridedSlice'while/gru_cell_1/ReadVariableOp:value:0-while/gru_cell_1/strided_slice/stack:output:0/while/gru_cell_1/strided_slice/stack_1:output:0/while/gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskŚ
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell_1/strided_slice:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_1ReadVariableOp*while_gru_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0w
&while/gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/gru_cell_1/strided_slice_1StridedSlice)while/gru_cell_1/ReadVariableOp_1:value:0/while/gru_cell_1/strided_slice_1/stack:output:01while/gru_cell_1/strided_slice_1/stack_1:output:01while/gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskŞ
while/gru_cell_1/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_1/strided_slice_1:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_2ReadVariableOp*while_gru_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0w
&while/gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        y
(while/gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/gru_cell_1/strided_slice_2StridedSlice)while/gru_cell_1/ReadVariableOp_2:value:0/while/gru_cell_1/strided_slice_2/stack:output:01while/gru_cell_1/strided_slice_2/stack_1:output:01while/gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskŞ
while/gru_cell_1/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_1/strided_slice_2:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_3ReadVariableOp,while_gru_cell_1_readvariableop_3_resource_0*
_output_shapes	
:*
dtype0p
&while/gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: s
(while/gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(while/gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ˇ
 while/gru_cell_1/strided_slice_3StridedSlice)while/gru_cell_1/ReadVariableOp_3:value:0/while/gru_cell_1/strided_slice_3/stack:output:01while/gru_cell_1/strided_slice_3/stack_1:output:01while/gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*

begin_mask
while/gru_cell_1/BiasAddBiasAdd!while/gru_cell_1/MatMul:product:0)while/gru_cell_1/strided_slice_3:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_4ReadVariableOp,while_gru_cell_1_readvariableop_3_resource_0*
_output_shapes	
:*
dtype0q
&while/gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:s
(while/gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(while/gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ľ
 while/gru_cell_1/strided_slice_4StridedSlice)while/gru_cell_1/ReadVariableOp_4:value:0/while/gru_cell_1/strided_slice_4/stack:output:01while/gru_cell_1/strided_slice_4/stack_1:output:01while/gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:
while/gru_cell_1/BiasAdd_1BiasAdd#while/gru_cell_1/MatMul_1:product:0)while/gru_cell_1/strided_slice_4:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_5ReadVariableOp,while_gru_cell_1_readvariableop_3_resource_0*
_output_shapes	
:*
dtype0q
&while/gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(while/gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(while/gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ľ
 while/gru_cell_1/strided_slice_5StridedSlice)while/gru_cell_1/ReadVariableOp_5:value:0/while/gru_cell_1/strided_slice_5/stack:output:01while/gru_cell_1/strided_slice_5/stack_1:output:01while/gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*
end_mask
while/gru_cell_1/BiasAdd_2BiasAdd#while/gru_cell_1/MatMul_2:product:0)while/gru_cell_1/strided_slice_5:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_6ReadVariableOp,while_gru_cell_1_readvariableop_6_resource_0* 
_output_shapes
:
*
dtype0w
&while/gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/gru_cell_1/strided_slice_6StridedSlice)while/gru_cell_1/ReadVariableOp_6:value:0/while/gru_cell_1/strided_slice_6/stack:output:01while/gru_cell_1/strided_slice_6/stack_1:output:01while/gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/gru_cell_1/MatMul_3MatMulwhile_placeholder_2)while/gru_cell_1/strided_slice_6:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_7ReadVariableOp,while_gru_cell_1_readvariableop_6_resource_0* 
_output_shapes
:
*
dtype0w
&while/gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/gru_cell_1/strided_slice_7StridedSlice)while/gru_cell_1/ReadVariableOp_7:value:0/while/gru_cell_1/strided_slice_7/stack:output:01while/gru_cell_1/strided_slice_7/stack_1:output:01while/gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/gru_cell_1/MatMul_4MatMulwhile_placeholder_2)while/gru_cell_1/strided_slice_7:output:0*
T0*
_output_shapes
:	
while/gru_cell_1/addAddV2!while/gru_cell_1/BiasAdd:output:0#while/gru_cell_1/MatMul_3:product:0*
T0*
_output_shapes
:	g
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*
_output_shapes
:	
while/gru_cell_1/add_1AddV2#while/gru_cell_1/BiasAdd_1:output:0#while/gru_cell_1/MatMul_4:product:0*
T0*
_output_shapes
:	k
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*
_output_shapes
:	z
while/gru_cell_1/mulMulwhile/gru_cell_1/Sigmoid_1:y:0while_placeholder_2*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_8ReadVariableOp,while_gru_cell_1_readvariableop_6_resource_0* 
_output_shapes
:
*
dtype0w
&while/gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        y
(while/gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/gru_cell_1/strided_slice_8StridedSlice)while/gru_cell_1/ReadVariableOp_8:value:0/while/gru_cell_1/strided_slice_8/stack:output:01while/gru_cell_1/strided_slice_8/stack_1:output:01while/gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/gru_cell_1/MatMul_5MatMulwhile/gru_cell_1/mul:z:0)while/gru_cell_1/strided_slice_8:output:0*
T0*
_output_shapes
:	
while/gru_cell_1/add_2AddV2#while/gru_cell_1/BiasAdd_2:output:0#while/gru_cell_1/MatMul_5:product:0*
T0*
_output_shapes
:	c
while/gru_cell_1/TanhTanhwhile/gru_cell_1/add_2:z:0*
T0*
_output_shapes
:	z
while/gru_cell_1/mul_1Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*
_output_shapes
:	[
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*
_output_shapes
:	|
while/gru_cell_1/mul_2Mulwhile/gru_cell_1/sub:z:0while/gru_cell_1/Tanh:y:0*
T0*
_output_shapes
:	
while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_1:z:0while/gru_cell_1/mul_2:z:0*
T0*
_output_shapes
:	Ă
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:éčŇM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éčŇo
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0^while/NoOp*
T0*
_output_shapes
:	

while/NoOpNoOp ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6"^while/gru_cell_1/ReadVariableOp_7"^while/gru_cell_1/ReadVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 "Z
*while_gru_cell_1_readvariableop_3_resource,while_gru_cell_1_readvariableop_3_resource_0"Z
*while_gru_cell_1_readvariableop_6_resource,while_gru_cell_1_readvariableop_6_resource_0"V
(while_gru_cell_1_readvariableop_resource*while_gru_cell_1_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*0
_input_shapes
: : : : :	: : : : : 2B
while/gru_cell_1/ReadVariableOpwhile/gru_cell_1/ReadVariableOp2F
!while/gru_cell_1/ReadVariableOp_1!while/gru_cell_1/ReadVariableOp_12F
!while/gru_cell_1/ReadVariableOp_2!while/gru_cell_1/ReadVariableOp_22F
!while/gru_cell_1/ReadVariableOp_3!while/gru_cell_1/ReadVariableOp_32F
!while/gru_cell_1/ReadVariableOp_4!while/gru_cell_1/ReadVariableOp_42F
!while/gru_cell_1/ReadVariableOp_5!while/gru_cell_1/ReadVariableOp_52F
!while/gru_cell_1/ReadVariableOp_6!while/gru_cell_1/ReadVariableOp_62F
!while/gru_cell_1/ReadVariableOp_7!while/gru_cell_1/ReadVariableOp_72F
!while/gru_cell_1/ReadVariableOp_8!while/gru_cell_1/ReadVariableOp_8: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: 
ć
Ä
H__inference_sequential_1_layer_call_and_return_conditional_losses_290613
embedding_1_input%
embedding_1_290595:	V 
gru_1_290598:

gru_1_290600:	 
gru_1_290602:

gru_1_290604:	!
dense_1_290607:	V
dense_1_290609:V
identity˘dense_1/StatefulPartitionedCall˘#embedding_1/StatefulPartitionedCall˘gru_1/StatefulPartitionedCallů
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallembedding_1_inputembedding_1_290595*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_embedding_1_layer_call_and_return_conditional_losses_289938ą
gru_1/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0gru_1_290598gru_1_290600gru_1_290602gru_1_290604*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_290474
dense_1/StatefulPartitionedCallStatefulPartitionedCall&gru_1/StatefulPartitionedCall:output:0dense_1_290607dense_1_290609*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙V*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_290203{
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙VŽ
NoOpNoOp ^dense_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:˙˙˙˙˙˙˙˙˙: : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:Z V
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameembedding_1_input
˝
Ś
while_body_289846
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_gru_cell_1_289868_0:
(
while_gru_cell_1_289870_0:	-
while_gru_cell_1_289872_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_gru_cell_1_289868:
&
while_gru_cell_1_289870:	+
while_gru_cell_1_289872:
˘(while/gru_cell_1/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	*
element_dtype0ň
(while/gru_cell_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_1_289868_0while_gru_cell_1_289870_0while_gru_cell_1_289872_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:	:	*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_289691Ú
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_1/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éčŇM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éčŇ
while/Identity_4Identity1while/gru_cell_1/StatefulPartitionedCall:output:1^while/NoOp*
T0*
_output_shapes
:	w

while/NoOpNoOp)^while/gru_cell_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "4
while_gru_cell_1_289868while_gru_cell_1_289868_0"4
while_gru_cell_1_289870while_gru_cell_1_289870_0"4
while_gru_cell_1_289872while_gru_cell_1_289872_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*0
_input_shapes
: : : : :	: : : : : 2T
(while/gru_cell_1/StatefulPartitionedCall(while/gru_cell_1/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: 
É

A__inference_gru_1_layer_call_and_return_conditional_losses_291915

inputs6
"gru_cell_1_readvariableop_resource:
3
$gru_cell_1_readvariableop_3_resource:	8
$gru_cell_1_readvariableop_6_resource:
>
+gru_cell_1_matmul_3_readvariableop_resource:	
identity˘AssignVariableOp˘ReadVariableOp˘"gru_cell_1/MatMul_3/ReadVariableOp˘"gru_cell_1/MatMul_4/ReadVariableOp˘gru_cell_1/ReadVariableOp˘gru_cell_1/ReadVariableOp_1˘gru_cell_1/ReadVariableOp_2˘gru_cell_1/ReadVariableOp_3˘gru_cell_1/ReadVariableOp_4˘gru_cell_1/ReadVariableOp_5˘gru_cell_1/ReadVariableOp_6˘gru_cell_1/ReadVariableOp_7˘gru_cell_1/ReadVariableOp_8˘gru_cell_1/mul/ReadVariableOp˘gru_cell_1/mul_1/ReadVariableOp˘whilec
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙˛
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:á
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_mask~
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype0o
gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        q
 gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       q
 gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¤
gru_cell_1/strided_sliceStridedSlice!gru_cell_1/ReadVariableOp:value:0'gru_cell_1/strided_slice/stack:output:0)gru_cell_1/strided_slice/stack_1:output:0)gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_cell_1/MatMulMatMulstrided_slice_1:output:0!gru_cell_1/strided_slice:output:0*
T0*
_output_shapes
:	
gru_cell_1/ReadVariableOp_1ReadVariableOp"gru_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype0q
 gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
gru_cell_1/strided_slice_1StridedSlice#gru_cell_1/ReadVariableOp_1:value:0)gru_cell_1/strided_slice_1/stack:output:0+gru_cell_1/strided_slice_1/stack_1:output:0+gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_cell_1/MatMul_1MatMulstrided_slice_1:output:0#gru_cell_1/strided_slice_1:output:0*
T0*
_output_shapes
:	
gru_cell_1/ReadVariableOp_2ReadVariableOp"gru_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype0q
 gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        s
"gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
gru_cell_1/strided_slice_2StridedSlice#gru_cell_1/ReadVariableOp_2:value:0)gru_cell_1/strided_slice_2/stack:output:0+gru_cell_1/strided_slice_2/stack_1:output:0+gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_cell_1/MatMul_2MatMulstrided_slice_1:output:0#gru_cell_1/strided_slice_2:output:0*
T0*
_output_shapes
:	}
gru_cell_1/ReadVariableOp_3ReadVariableOp$gru_cell_1_readvariableop_3_resource*
_output_shapes	
:*
dtype0j
 gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: m
"gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_cell_1/strided_slice_3StridedSlice#gru_cell_1/ReadVariableOp_3:value:0)gru_cell_1/strided_slice_3/stack:output:0+gru_cell_1/strided_slice_3/stack_1:output:0+gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*

begin_mask
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0#gru_cell_1/strided_slice_3:output:0*
T0*
_output_shapes
:	}
gru_cell_1/ReadVariableOp_4ReadVariableOp$gru_cell_1_readvariableop_3_resource*
_output_shapes	
:*
dtype0k
 gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:m
"gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_cell_1/strided_slice_4StridedSlice#gru_cell_1/ReadVariableOp_4:value:0)gru_cell_1/strided_slice_4/stack:output:0+gru_cell_1/strided_slice_4/stack_1:output:0+gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0#gru_cell_1/strided_slice_4:output:0*
T0*
_output_shapes
:	}
gru_cell_1/ReadVariableOp_5ReadVariableOp$gru_cell_1_readvariableop_3_resource*
_output_shapes	
:*
dtype0k
 gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:l
"gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: l
"gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_cell_1/strided_slice_5StridedSlice#gru_cell_1/ReadVariableOp_5:value:0)gru_cell_1/strided_slice_5/stack:output:0+gru_cell_1/strided_slice_5/stack_1:output:0+gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*
end_mask
gru_cell_1/BiasAdd_2BiasAddgru_cell_1/MatMul_2:product:0#gru_cell_1/strided_slice_5:output:0*
T0*
_output_shapes
:	
gru_cell_1/ReadVariableOp_6ReadVariableOp$gru_cell_1_readvariableop_6_resource* 
_output_shapes
:
*
dtype0q
 gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
gru_cell_1/strided_slice_6StridedSlice#gru_cell_1/ReadVariableOp_6:value:0)gru_cell_1/strided_slice_6/stack:output:0+gru_cell_1/strided_slice_6/stack_1:output:0+gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
"gru_cell_1/MatMul_3/ReadVariableOpReadVariableOp+gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_1/MatMul_3MatMul*gru_cell_1/MatMul_3/ReadVariableOp:value:0#gru_cell_1/strided_slice_6:output:0*
T0*
_output_shapes
:	
gru_cell_1/ReadVariableOp_7ReadVariableOp$gru_cell_1_readvariableop_6_resource* 
_output_shapes
:
*
dtype0q
 gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
gru_cell_1/strided_slice_7StridedSlice#gru_cell_1/ReadVariableOp_7:value:0)gru_cell_1/strided_slice_7/stack:output:0+gru_cell_1/strided_slice_7/stack_1:output:0+gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
"gru_cell_1/MatMul_4/ReadVariableOpReadVariableOp+gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_1/MatMul_4MatMul*gru_cell_1/MatMul_4/ReadVariableOp:value:0#gru_cell_1/strided_slice_7:output:0*
T0*
_output_shapes
:	}
gru_cell_1/addAddV2gru_cell_1/BiasAdd:output:0gru_cell_1/MatMul_3:product:0*
T0*
_output_shapes
:	[
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*
_output_shapes
:	
gru_cell_1/add_1AddV2gru_cell_1/BiasAdd_1:output:0gru_cell_1/MatMul_4:product:0*
T0*
_output_shapes
:	_
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*
_output_shapes
:	
gru_cell_1/mul/ReadVariableOpReadVariableOp+gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0%gru_cell_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:	
gru_cell_1/ReadVariableOp_8ReadVariableOp$gru_cell_1_readvariableop_6_resource* 
_output_shapes
:
*
dtype0q
 gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        s
"gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
gru_cell_1/strided_slice_8StridedSlice#gru_cell_1/ReadVariableOp_8:value:0)gru_cell_1/strided_slice_8/stack:output:0+gru_cell_1/strided_slice_8/stack_1:output:0+gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_cell_1/MatMul_5MatMulgru_cell_1/mul:z:0#gru_cell_1/strided_slice_8:output:0*
T0*
_output_shapes
:	
gru_cell_1/add_2AddV2gru_cell_1/BiasAdd_2:output:0gru_cell_1/MatMul_5:product:0*
T0*
_output_shapes
:	W
gru_cell_1/TanhTanhgru_cell_1/add_2:z:0*
T0*
_output_shapes
:	
gru_cell_1/mul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0'gru_cell_1/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	U
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?r
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*
_output_shapes
:	j
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Tanh:y:0*
T0*
_output_shapes
:	o
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*
_output_shapes
:	n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ś
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : {
ReadVariableOpReadVariableOp+gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ľ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0ReadVariableOp:value:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_1_readvariableop_resource$gru_cell_1_readvariableop_3_resource$gru_cell_1_readvariableop_6_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_291790*
condR
while_cond_291789*0
output_shapes
: : : : :	: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ă
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:˙
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    Ą
AssignVariableOpAssignVariableOp+gru_cell_1_matmul_3_readvariableop_resourcewhile:output:4^ReadVariableOp#^gru_cell_1/MatMul_3/ReadVariableOp#^gru_cell_1/MatMul_4/ReadVariableOp^gru_cell_1/mul/ReadVariableOp ^gru_cell_1/mul_1/ReadVariableOp*
_output_shapes
 *
dtype0c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp^AssignVariableOp^ReadVariableOp#^gru_cell_1/MatMul_3/ReadVariableOp#^gru_cell_1/MatMul_4/ReadVariableOp^gru_cell_1/ReadVariableOp^gru_cell_1/ReadVariableOp_1^gru_cell_1/ReadVariableOp_2^gru_cell_1/ReadVariableOp_3^gru_cell_1/ReadVariableOp_4^gru_cell_1/ReadVariableOp_5^gru_cell_1/ReadVariableOp_6^gru_cell_1/ReadVariableOp_7^gru_cell_1/ReadVariableOp_8^gru_cell_1/mul/ReadVariableOp ^gru_cell_1/mul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙: : : : 2$
AssignVariableOpAssignVariableOp2 
ReadVariableOpReadVariableOp2H
"gru_cell_1/MatMul_3/ReadVariableOp"gru_cell_1/MatMul_3/ReadVariableOp2H
"gru_cell_1/MatMul_4/ReadVariableOp"gru_cell_1/MatMul_4/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2:
gru_cell_1/ReadVariableOp_1gru_cell_1/ReadVariableOp_12:
gru_cell_1/ReadVariableOp_2gru_cell_1/ReadVariableOp_22:
gru_cell_1/ReadVariableOp_3gru_cell_1/ReadVariableOp_32:
gru_cell_1/ReadVariableOp_4gru_cell_1/ReadVariableOp_42:
gru_cell_1/ReadVariableOp_5gru_cell_1/ReadVariableOp_52:
gru_cell_1/ReadVariableOp_6gru_cell_1/ReadVariableOp_62:
gru_cell_1/ReadVariableOp_7gru_cell_1/ReadVariableOp_72:
gru_cell_1/ReadVariableOp_8gru_cell_1/ReadVariableOp_82>
gru_cell_1/mul/ReadVariableOpgru_cell_1/mul/ReadVariableOp2B
gru_cell_1/mul_1/ReadVariableOpgru_cell_1/mul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ú 
ó
"__inference__traced_restore_292619
file_prefix:
'assignvariableop_embedding_1_embeddings:	V4
!assignvariableop_1_dense_1_kernel:	V-
assignvariableop_2_dense_1_bias:V>
*assignvariableop_3_gru_1_gru_cell_1_kernel:
H
4assignvariableop_4_gru_1_gru_cell_1_recurrent_kernel:
7
(assignvariableop_5_gru_1_gru_cell_1_bias:	4
!assignvariableop_6_gru_1_variable:	

identity_8˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_2˘AssignVariableOp_3˘AssignVariableOp_4˘AssignVariableOp_5˘AssignVariableOp_6ó
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-1/keras_api/states/0/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B Ć
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*4
_output_shapes"
 ::::::::*
dtypes

2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp'assignvariableop_embedding_1_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_1_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_1_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp*assignvariableop_3_gru_1_gru_cell_1_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Ł
AssignVariableOp_4AssignVariableOp4assignvariableop_4_gru_1_gru_cell_1_recurrent_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp(assignvariableop_5_gru_1_gru_cell_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp!assignvariableop_6_gru_1_variableIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ë

Identity_7Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_8IdentityIdentity_7:output:0^NoOp_1*
T0*
_output_shapes
: Ů
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6*"
_acd_function_control_output(*
_output_shapes
 "!

identity_8Identity_8:output:0*#
_input_shapes
: : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_6:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
/
Ç
A__inference_gru_1_layer_call_and_return_conditional_losses_289910

inputs$
gru_cell_1_289831:	%
gru_cell_1_289833:
 
gru_cell_1_289835:	%
gru_cell_1_289837:

identity˘AssignVariableOp˘ReadVariableOp˘"gru_cell_1/StatefulPartitionedCall˘whilec
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙˛
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:á
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_maskť
"gru_cell_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0gru_cell_1_289831gru_cell_1_289833gru_cell_1_289835gru_cell_1_289837*
Tin	
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:	:	*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_289799n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ś
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : a
ReadVariableOpReadVariableOpgru_cell_1_289831*
_output_shapes
:	*
dtype0c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : î
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0ReadVariableOp:value:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_1_289833gru_cell_1_289835gru_cell_1_289837*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_289846*
condR
while_cond_289845*0
output_shapes
: : : : :	: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ă
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:˙
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *     
AssignVariableOpAssignVariableOpgru_cell_1_289831while:output:4^ReadVariableOp#^gru_cell_1/StatefulPartitionedCall*
_output_shapes
 *
dtype0c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp^AssignVariableOp^ReadVariableOp#^gru_cell_1/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙: : : : 2$
AssignVariableOpAssignVariableOp2 
ReadVariableOpReadVariableOp2H
"gru_cell_1/StatefulPartitionedCall"gru_cell_1/StatefulPartitionedCall2
whilewhile:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ď
ű
C__inference_dense_1_layer_call_and_return_conditional_losses_290203

inputs4
!tensordot_readvariableop_resource:	V-
biasadd_readvariableop_resource:V
identity˘BiasAdd/ReadVariableOp˘Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	V*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ť
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ż
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙V[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:VY
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙Vr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:V*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙Vc
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙Vz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ť
	
H__inference_sequential_1_layer_call_and_return_conditional_losses_291159

inputs6
#embedding_1_embedding_lookup_290909:	V<
(gru_1_gru_cell_1_readvariableop_resource:
9
*gru_1_gru_cell_1_readvariableop_3_resource:	>
*gru_1_gru_cell_1_readvariableop_6_resource:
D
1gru_1_gru_cell_1_matmul_3_readvariableop_resource:	<
)dense_1_tensordot_readvariableop_resource:	V5
'dense_1_biasadd_readvariableop_resource:V
identity˘dense_1/BiasAdd/ReadVariableOp˘ dense_1/Tensordot/ReadVariableOp˘embedding_1/embedding_lookup˘gru_1/AssignVariableOp˘gru_1/ReadVariableOp˘(gru_1/gru_cell_1/MatMul_3/ReadVariableOp˘(gru_1/gru_cell_1/MatMul_4/ReadVariableOp˘gru_1/gru_cell_1/ReadVariableOp˘!gru_1/gru_cell_1/ReadVariableOp_1˘!gru_1/gru_cell_1/ReadVariableOp_2˘!gru_1/gru_cell_1/ReadVariableOp_3˘!gru_1/gru_cell_1/ReadVariableOp_4˘!gru_1/gru_cell_1/ReadVariableOp_5˘!gru_1/gru_cell_1/ReadVariableOp_6˘!gru_1/gru_cell_1/ReadVariableOp_7˘!gru_1/gru_cell_1/ReadVariableOp_8˘#gru_1/gru_cell_1/mul/ReadVariableOp˘%gru_1/gru_cell_1/mul_1/ReadVariableOp˘gru_1/whilea
embedding_1/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ě
embedding_1/embedding_lookupResourceGather#embedding_1_embedding_lookup_290909embedding_1/Cast:y:0*
Tindices0*6
_class,
*(loc:@embedding_1/embedding_lookup/290909*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0Ç
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_1/embedding_lookup/290909*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙i
gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¤
gru_1/transpose	Transpose0embedding_1/embedding_lookup/Identity_1:output:0gru_1/transpose/perm:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙N
gru_1/ShapeShapegru_1/transpose:y:0*
T0*
_output_shapes
:c
gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ď
gru_1/strided_sliceStridedSlicegru_1/Shape:output:0"gru_1/strided_slice/stack:output:0$gru_1/strided_slice/stack_1:output:0$gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ä
gru_1/TensorArrayV2TensorListReserve*gru_1/TensorArrayV2/element_shape:output:0gru_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
;gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ň
-gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_1/transpose:y:0Dgru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇe
gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:˙
gru_1/strided_slice_1StridedSlicegru_1/transpose:y:0$gru_1/strided_slice_1/stack:output:0&gru_1/strided_slice_1/stack_1:output:0&gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_mask
gru_1/gru_cell_1/ReadVariableOpReadVariableOp(gru_1_gru_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype0u
$gru_1/gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        w
&gru_1/gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       w
&gru_1/gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Â
gru_1/gru_cell_1/strided_sliceStridedSlice'gru_1/gru_cell_1/ReadVariableOp:value:0-gru_1/gru_cell_1/strided_slice/stack:output:0/gru_1/gru_cell_1/strided_slice/stack_1:output:0/gru_1/gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_1/gru_cell_1/MatMulMatMulgru_1/strided_slice_1:output:0'gru_1/gru_cell_1/strided_slice:output:0*
T0*
_output_shapes
:	
!gru_1/gru_cell_1/ReadVariableOp_1ReadVariableOp(gru_1_gru_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype0w
&gru_1/gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       y
(gru_1/gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(gru_1/gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 gru_1/gru_cell_1/strided_slice_1StridedSlice)gru_1/gru_cell_1/ReadVariableOp_1:value:0/gru_1/gru_cell_1/strided_slice_1/stack:output:01gru_1/gru_cell_1/strided_slice_1/stack_1:output:01gru_1/gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_1/gru_cell_1/MatMul_1MatMulgru_1/strided_slice_1:output:0)gru_1/gru_cell_1/strided_slice_1:output:0*
T0*
_output_shapes
:	
!gru_1/gru_cell_1/ReadVariableOp_2ReadVariableOp(gru_1_gru_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype0w
&gru_1/gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       y
(gru_1/gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        y
(gru_1/gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 gru_1/gru_cell_1/strided_slice_2StridedSlice)gru_1/gru_cell_1/ReadVariableOp_2:value:0/gru_1/gru_cell_1/strided_slice_2/stack:output:01gru_1/gru_cell_1/strided_slice_2/stack_1:output:01gru_1/gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_1/gru_cell_1/MatMul_2MatMulgru_1/strided_slice_1:output:0)gru_1/gru_cell_1/strided_slice_2:output:0*
T0*
_output_shapes
:	
!gru_1/gru_cell_1/ReadVariableOp_3ReadVariableOp*gru_1_gru_cell_1_readvariableop_3_resource*
_output_shapes	
:*
dtype0p
&gru_1/gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: s
(gru_1/gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(gru_1/gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ˇ
 gru_1/gru_cell_1/strided_slice_3StridedSlice)gru_1/gru_cell_1/ReadVariableOp_3:value:0/gru_1/gru_cell_1/strided_slice_3/stack:output:01gru_1/gru_cell_1/strided_slice_3/stack_1:output:01gru_1/gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*

begin_mask
gru_1/gru_cell_1/BiasAddBiasAdd!gru_1/gru_cell_1/MatMul:product:0)gru_1/gru_cell_1/strided_slice_3:output:0*
T0*
_output_shapes
:	
!gru_1/gru_cell_1/ReadVariableOp_4ReadVariableOp*gru_1_gru_cell_1_readvariableop_3_resource*
_output_shapes	
:*
dtype0q
&gru_1/gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:s
(gru_1/gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(gru_1/gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ľ
 gru_1/gru_cell_1/strided_slice_4StridedSlice)gru_1/gru_cell_1/ReadVariableOp_4:value:0/gru_1/gru_cell_1/strided_slice_4/stack:output:01gru_1/gru_cell_1/strided_slice_4/stack_1:output:01gru_1/gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:
gru_1/gru_cell_1/BiasAdd_1BiasAdd#gru_1/gru_cell_1/MatMul_1:product:0)gru_1/gru_cell_1/strided_slice_4:output:0*
T0*
_output_shapes
:	
!gru_1/gru_cell_1/ReadVariableOp_5ReadVariableOp*gru_1_gru_cell_1_readvariableop_3_resource*
_output_shapes	
:*
dtype0q
&gru_1/gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(gru_1/gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(gru_1/gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ľ
 gru_1/gru_cell_1/strided_slice_5StridedSlice)gru_1/gru_cell_1/ReadVariableOp_5:value:0/gru_1/gru_cell_1/strided_slice_5/stack:output:01gru_1/gru_cell_1/strided_slice_5/stack_1:output:01gru_1/gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*
end_mask
gru_1/gru_cell_1/BiasAdd_2BiasAdd#gru_1/gru_cell_1/MatMul_2:product:0)gru_1/gru_cell_1/strided_slice_5:output:0*
T0*
_output_shapes
:	
!gru_1/gru_cell_1/ReadVariableOp_6ReadVariableOp*gru_1_gru_cell_1_readvariableop_6_resource* 
_output_shapes
:
*
dtype0w
&gru_1/gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(gru_1/gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(gru_1/gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 gru_1/gru_cell_1/strided_slice_6StridedSlice)gru_1/gru_cell_1/ReadVariableOp_6:value:0/gru_1/gru_cell_1/strided_slice_6/stack:output:01gru_1/gru_cell_1/strided_slice_6/stack_1:output:01gru_1/gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
(gru_1/gru_cell_1/MatMul_3/ReadVariableOpReadVariableOp1gru_1_gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0Ş
gru_1/gru_cell_1/MatMul_3MatMul0gru_1/gru_cell_1/MatMul_3/ReadVariableOp:value:0)gru_1/gru_cell_1/strided_slice_6:output:0*
T0*
_output_shapes
:	
!gru_1/gru_cell_1/ReadVariableOp_7ReadVariableOp*gru_1_gru_cell_1_readvariableop_6_resource* 
_output_shapes
:
*
dtype0w
&gru_1/gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       y
(gru_1/gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(gru_1/gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 gru_1/gru_cell_1/strided_slice_7StridedSlice)gru_1/gru_cell_1/ReadVariableOp_7:value:0/gru_1/gru_cell_1/strided_slice_7/stack:output:01gru_1/gru_cell_1/strided_slice_7/stack_1:output:01gru_1/gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
(gru_1/gru_cell_1/MatMul_4/ReadVariableOpReadVariableOp1gru_1_gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0Ş
gru_1/gru_cell_1/MatMul_4MatMul0gru_1/gru_cell_1/MatMul_4/ReadVariableOp:value:0)gru_1/gru_cell_1/strided_slice_7:output:0*
T0*
_output_shapes
:	
gru_1/gru_cell_1/addAddV2!gru_1/gru_cell_1/BiasAdd:output:0#gru_1/gru_cell_1/MatMul_3:product:0*
T0*
_output_shapes
:	g
gru_1/gru_cell_1/SigmoidSigmoidgru_1/gru_cell_1/add:z:0*
T0*
_output_shapes
:	
gru_1/gru_cell_1/add_1AddV2#gru_1/gru_cell_1/BiasAdd_1:output:0#gru_1/gru_cell_1/MatMul_4:product:0*
T0*
_output_shapes
:	k
gru_1/gru_cell_1/Sigmoid_1Sigmoidgru_1/gru_cell_1/add_1:z:0*
T0*
_output_shapes
:	
#gru_1/gru_cell_1/mul/ReadVariableOpReadVariableOp1gru_1_gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_1/gru_cell_1/mulMulgru_1/gru_cell_1/Sigmoid_1:y:0+gru_1/gru_cell_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:	
!gru_1/gru_cell_1/ReadVariableOp_8ReadVariableOp*gru_1_gru_cell_1_readvariableop_6_resource* 
_output_shapes
:
*
dtype0w
&gru_1/gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       y
(gru_1/gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        y
(gru_1/gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 gru_1/gru_cell_1/strided_slice_8StridedSlice)gru_1/gru_cell_1/ReadVariableOp_8:value:0/gru_1/gru_cell_1/strided_slice_8/stack:output:01gru_1/gru_cell_1/strided_slice_8/stack_1:output:01gru_1/gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_1/gru_cell_1/MatMul_5MatMulgru_1/gru_cell_1/mul:z:0)gru_1/gru_cell_1/strided_slice_8:output:0*
T0*
_output_shapes
:	
gru_1/gru_cell_1/add_2AddV2#gru_1/gru_cell_1/BiasAdd_2:output:0#gru_1/gru_cell_1/MatMul_5:product:0*
T0*
_output_shapes
:	c
gru_1/gru_cell_1/TanhTanhgru_1/gru_cell_1/add_2:z:0*
T0*
_output_shapes
:	
%gru_1/gru_cell_1/mul_1/ReadVariableOpReadVariableOp1gru_1_gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_1/gru_cell_1/mul_1Mulgru_1/gru_cell_1/Sigmoid:y:0-gru_1/gru_cell_1/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	[
gru_1/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_1/gru_cell_1/subSubgru_1/gru_cell_1/sub/x:output:0gru_1/gru_cell_1/Sigmoid:y:0*
T0*
_output_shapes
:	|
gru_1/gru_cell_1/mul_2Mulgru_1/gru_cell_1/sub:z:0gru_1/gru_cell_1/Tanh:y:0*
T0*
_output_shapes
:	
gru_1/gru_cell_1/add_3AddV2gru_1/gru_cell_1/mul_1:z:0gru_1/gru_cell_1/mul_2:z:0*
T0*
_output_shapes
:	t
#gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      Č
gru_1/TensorArrayV2_1TensorListReserve,gru_1/TensorArrayV2_1/element_shape:output:0gru_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇL

gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 
gru_1/ReadVariableOpReadVariableOp1gru_1_gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0i
gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Z
gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ó
gru_1/whileWhile!gru_1/while/loop_counter:output:0'gru_1/while/maximum_iterations:output:0gru_1/time:output:0gru_1/TensorArrayV2_1:handle:0gru_1/ReadVariableOp:value:0gru_1/strided_slice:output:0=gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0(gru_1_gru_cell_1_readvariableop_resource*gru_1_gru_cell_1_readvariableop_3_resource*gru_1_gru_cell_1_readvariableop_6_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *#
bodyR
gru_1_while_body_291008*#
condR
gru_1_while_cond_291007*0
output_shapes
: : : : :	: : : : : *
parallel_iterations 
6gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ő
(gru_1/TensorArrayV2Stack/TensorListStackTensorListStackgru_1/while:output:3?gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0n
gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙g
gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_1/strided_slice_2StridedSlice1gru_1/TensorArrayV2Stack/TensorListStack:tensor:0$gru_1/strided_slice_2/stack:output:0&gru_1/strided_slice_2/stack_1:output:0&gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_maskk
gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Š
gru_1/transpose_1	Transpose1gru_1/TensorArrayV2Stack/TensorListStack:tensor:0gru_1/transpose_1/perm:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙a
gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    Ń
gru_1/AssignVariableOpAssignVariableOp1gru_1_gru_cell_1_matmul_3_readvariableop_resourcegru_1/while:output:4^gru_1/ReadVariableOp)^gru_1/gru_cell_1/MatMul_3/ReadVariableOp)^gru_1/gru_cell_1/MatMul_4/ReadVariableOp$^gru_1/gru_cell_1/mul/ReadVariableOp&^gru_1/gru_cell_1/mul_1/ReadVariableOp*
_output_shapes
 *
dtype0
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes
:	V*
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       \
dense_1/Tensordot/ShapeShapegru_1/transpose_1:y:0*
T0*
_output_shapes
:a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ű
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ź
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_1/Tensordot/transpose	Transposegru_1/transpose_1:y:0!dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˘
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Vc
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Va
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙V
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:V*
dtype0
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙Vk
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙VÍ
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp^embedding_1/embedding_lookup^gru_1/AssignVariableOp^gru_1/ReadVariableOp)^gru_1/gru_cell_1/MatMul_3/ReadVariableOp)^gru_1/gru_cell_1/MatMul_4/ReadVariableOp ^gru_1/gru_cell_1/ReadVariableOp"^gru_1/gru_cell_1/ReadVariableOp_1"^gru_1/gru_cell_1/ReadVariableOp_2"^gru_1/gru_cell_1/ReadVariableOp_3"^gru_1/gru_cell_1/ReadVariableOp_4"^gru_1/gru_cell_1/ReadVariableOp_5"^gru_1/gru_cell_1/ReadVariableOp_6"^gru_1/gru_cell_1/ReadVariableOp_7"^gru_1/gru_cell_1/ReadVariableOp_8$^gru_1/gru_cell_1/mul/ReadVariableOp&^gru_1/gru_cell_1/mul_1/ReadVariableOp^gru_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:˙˙˙˙˙˙˙˙˙: : : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup20
gru_1/AssignVariableOpgru_1/AssignVariableOp2,
gru_1/ReadVariableOpgru_1/ReadVariableOp2T
(gru_1/gru_cell_1/MatMul_3/ReadVariableOp(gru_1/gru_cell_1/MatMul_3/ReadVariableOp2T
(gru_1/gru_cell_1/MatMul_4/ReadVariableOp(gru_1/gru_cell_1/MatMul_4/ReadVariableOp2B
gru_1/gru_cell_1/ReadVariableOpgru_1/gru_cell_1/ReadVariableOp2F
!gru_1/gru_cell_1/ReadVariableOp_1!gru_1/gru_cell_1/ReadVariableOp_12F
!gru_1/gru_cell_1/ReadVariableOp_2!gru_1/gru_cell_1/ReadVariableOp_22F
!gru_1/gru_cell_1/ReadVariableOp_3!gru_1/gru_cell_1/ReadVariableOp_32F
!gru_1/gru_cell_1/ReadVariableOp_4!gru_1/gru_cell_1/ReadVariableOp_42F
!gru_1/gru_cell_1/ReadVariableOp_5!gru_1/gru_cell_1/ReadVariableOp_52F
!gru_1/gru_cell_1/ReadVariableOp_6!gru_1/gru_cell_1/ReadVariableOp_62F
!gru_1/gru_cell_1/ReadVariableOp_7!gru_1/gru_cell_1/ReadVariableOp_72F
!gru_1/gru_cell_1/ReadVariableOp_8!gru_1/gru_cell_1/ReadVariableOp_82J
#gru_1/gru_cell_1/mul/ReadVariableOp#gru_1/gru_cell_1/mul/ReadVariableOp2N
%gru_1/gru_cell_1/mul_1/ReadVariableOp%gru_1/gru_cell_1/mul_1/ReadVariableOp2
gru_1/whilegru_1/while:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ľH
˛
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_292464

inputs
states_0+
readvariableop_resource:
(
readvariableop_3_resource:	-
readvariableop_6_resource:

identity

identity_1˘ReadVariableOp˘ReadVariableOp_1˘ReadVariableOp_2˘ReadVariableOp_3˘ReadVariableOp_4˘ReadVariableOp_5˘ReadVariableOp_6˘ReadVariableOp_7˘ReadVariableOp_8h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      í
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskZ
MatMulMatMulinputsstrided_slice:output:0*
T0*
_output_shapes
:	j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask^
MatMul_1MatMulinputsstrided_slice_1:output:0*
T0*
_output_shapes
:	j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask^
MatMul_2MatMulinputsstrided_slice_2:output:0*
T0*
_output_shapes
:	g
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes	
:*
dtype0_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: b
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*

begin_maskh
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*
_output_shapes
:	g
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes	
:*
dtype0`
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Đ
strided_slice_4StridedSliceReadVariableOp_4:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:l
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*
_output_shapes
:	g
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes	
:*
dtype0`
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ŕ
strided_slice_5StridedSliceReadVariableOp_5:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*
end_maskl
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*
_output_shapes
:	l
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource* 
_output_shapes
:
*
dtype0f
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_6StridedSliceReadVariableOp_6:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask`
MatMul_3MatMulstates_0strided_slice_6:output:0*
T0*
_output_shapes
:	l
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource* 
_output_shapes
:
*
dtype0f
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_7StridedSliceReadVariableOp_7:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask`
MatMul_4MatMulstates_0strided_slice_7:output:0*
T0*
_output_shapes
:	\
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*
_output_shapes
:	E
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:	`
add_1AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*
_output_shapes
:	I
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:	M
mulMulSigmoid_1:y:0states_0*
T0*
_output_shapes
:	l
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource* 
_output_shapes
:
*
dtype0f
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_8StridedSliceReadVariableOp_8:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask_
MatMul_5MatMulmul:z:0strided_slice_8:output:0*
T0*
_output_shapes
:	`
add_2AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*
_output_shapes
:	A
TanhTanh	add_2:z:0*
T0*
_output_shapes
:	M
mul_1MulSigmoid:y:0states_0*
T0*
_output_shapes
:	J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Q
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:	I
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:	N
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:	P
IdentityIdentity	add_3:z:0^NoOp*
T0*
_output_shapes
:	R

Identity_1Identity	add_3:z:0^NoOp*
T0*
_output_shapes
:	ď
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:	:	: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_8:G C

_output_shapes
:	
 
_user_specified_nameinputs:IE

_output_shapes
:	
"
_user_specified_name
states/0
´	
Ż
-__inference_sequential_1_layer_call_fn_290632

inputs
unknown:	V
	unknown_0:

	unknown_1:	
	unknown_2:

	unknown_3:	
	unknown_4:	V
	unknown_5:V
identity˘StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙V*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_290210s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙V`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:˙˙˙˙˙˙˙˙˙: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
˝
Ś
while_body_289448
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_gru_cell_1_289545_0:
(
while_gru_cell_1_289547_0:	-
while_gru_cell_1_289549_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_gru_cell_1_289545:
&
while_gru_cell_1_289547:	+
while_gru_cell_1_289549:
˘(while/gru_cell_1/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	*
element_dtype0ň
(while/gru_cell_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_1_289545_0while_gru_cell_1_289547_0while_gru_cell_1_289549_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:	:	*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_289544Ú
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_1/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éčŇM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éčŇ
while/Identity_4Identity1while/gru_cell_1/StatefulPartitionedCall:output:1^while/NoOp*
T0*
_output_shapes
:	w

while/NoOpNoOp)^while/gru_cell_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "4
while_gru_cell_1_289545while_gru_cell_1_289545_0"4
while_gru_cell_1_289547while_gru_cell_1_289547_0"4
while_gru_cell_1_289549while_gru_cell_1_289549_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*0
_input_shapes
: : : : :	: : : : : 2T
(while/gru_cell_1/StatefulPartitionedCall(while/gru_cell_1/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: 
çv
Ť	
while_body_291790
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
*while_gru_cell_1_readvariableop_resource_0:
;
,while_gru_cell_1_readvariableop_3_resource_0:	@
,while_gru_cell_1_readvariableop_6_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
(while_gru_cell_1_readvariableop_resource:
9
*while_gru_cell_1_readvariableop_3_resource:	>
*while_gru_cell_1_readvariableop_6_resource:
˘while/gru_cell_1/ReadVariableOp˘!while/gru_cell_1/ReadVariableOp_1˘!while/gru_cell_1/ReadVariableOp_2˘!while/gru_cell_1/ReadVariableOp_3˘!while/gru_cell_1/ReadVariableOp_4˘!while/gru_cell_1/ReadVariableOp_5˘!while/gru_cell_1/ReadVariableOp_6˘!while/gru_cell_1/ReadVariableOp_7˘!while/gru_cell_1/ReadVariableOp_8
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	*
element_dtype0
while/gru_cell_1/ReadVariableOpReadVariableOp*while_gru_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0u
$while/gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        w
&while/gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       w
&while/gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Â
while/gru_cell_1/strided_sliceStridedSlice'while/gru_cell_1/ReadVariableOp:value:0-while/gru_cell_1/strided_slice/stack:output:0/while/gru_cell_1/strided_slice/stack_1:output:0/while/gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskŚ
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell_1/strided_slice:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_1ReadVariableOp*while_gru_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0w
&while/gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/gru_cell_1/strided_slice_1StridedSlice)while/gru_cell_1/ReadVariableOp_1:value:0/while/gru_cell_1/strided_slice_1/stack:output:01while/gru_cell_1/strided_slice_1/stack_1:output:01while/gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskŞ
while/gru_cell_1/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_1/strided_slice_1:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_2ReadVariableOp*while_gru_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0w
&while/gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        y
(while/gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/gru_cell_1/strided_slice_2StridedSlice)while/gru_cell_1/ReadVariableOp_2:value:0/while/gru_cell_1/strided_slice_2/stack:output:01while/gru_cell_1/strided_slice_2/stack_1:output:01while/gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskŞ
while/gru_cell_1/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_1/strided_slice_2:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_3ReadVariableOp,while_gru_cell_1_readvariableop_3_resource_0*
_output_shapes	
:*
dtype0p
&while/gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: s
(while/gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(while/gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ˇ
 while/gru_cell_1/strided_slice_3StridedSlice)while/gru_cell_1/ReadVariableOp_3:value:0/while/gru_cell_1/strided_slice_3/stack:output:01while/gru_cell_1/strided_slice_3/stack_1:output:01while/gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*

begin_mask
while/gru_cell_1/BiasAddBiasAdd!while/gru_cell_1/MatMul:product:0)while/gru_cell_1/strided_slice_3:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_4ReadVariableOp,while_gru_cell_1_readvariableop_3_resource_0*
_output_shapes	
:*
dtype0q
&while/gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:s
(while/gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(while/gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ľ
 while/gru_cell_1/strided_slice_4StridedSlice)while/gru_cell_1/ReadVariableOp_4:value:0/while/gru_cell_1/strided_slice_4/stack:output:01while/gru_cell_1/strided_slice_4/stack_1:output:01while/gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:
while/gru_cell_1/BiasAdd_1BiasAdd#while/gru_cell_1/MatMul_1:product:0)while/gru_cell_1/strided_slice_4:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_5ReadVariableOp,while_gru_cell_1_readvariableop_3_resource_0*
_output_shapes	
:*
dtype0q
&while/gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(while/gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(while/gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ľ
 while/gru_cell_1/strided_slice_5StridedSlice)while/gru_cell_1/ReadVariableOp_5:value:0/while/gru_cell_1/strided_slice_5/stack:output:01while/gru_cell_1/strided_slice_5/stack_1:output:01while/gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*
end_mask
while/gru_cell_1/BiasAdd_2BiasAdd#while/gru_cell_1/MatMul_2:product:0)while/gru_cell_1/strided_slice_5:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_6ReadVariableOp,while_gru_cell_1_readvariableop_6_resource_0* 
_output_shapes
:
*
dtype0w
&while/gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/gru_cell_1/strided_slice_6StridedSlice)while/gru_cell_1/ReadVariableOp_6:value:0/while/gru_cell_1/strided_slice_6/stack:output:01while/gru_cell_1/strided_slice_6/stack_1:output:01while/gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/gru_cell_1/MatMul_3MatMulwhile_placeholder_2)while/gru_cell_1/strided_slice_6:output:0*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_7ReadVariableOp,while_gru_cell_1_readvariableop_6_resource_0* 
_output_shapes
:
*
dtype0w
&while/gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/gru_cell_1/strided_slice_7StridedSlice)while/gru_cell_1/ReadVariableOp_7:value:0/while/gru_cell_1/strided_slice_7/stack:output:01while/gru_cell_1/strided_slice_7/stack_1:output:01while/gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/gru_cell_1/MatMul_4MatMulwhile_placeholder_2)while/gru_cell_1/strided_slice_7:output:0*
T0*
_output_shapes
:	
while/gru_cell_1/addAddV2!while/gru_cell_1/BiasAdd:output:0#while/gru_cell_1/MatMul_3:product:0*
T0*
_output_shapes
:	g
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*
_output_shapes
:	
while/gru_cell_1/add_1AddV2#while/gru_cell_1/BiasAdd_1:output:0#while/gru_cell_1/MatMul_4:product:0*
T0*
_output_shapes
:	k
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*
_output_shapes
:	z
while/gru_cell_1/mulMulwhile/gru_cell_1/Sigmoid_1:y:0while_placeholder_2*
T0*
_output_shapes
:	
!while/gru_cell_1/ReadVariableOp_8ReadVariableOp,while_gru_cell_1_readvariableop_6_resource_0* 
_output_shapes
:
*
dtype0w
&while/gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       y
(while/gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        y
(while/gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/gru_cell_1/strided_slice_8StridedSlice)while/gru_cell_1/ReadVariableOp_8:value:0/while/gru_cell_1/strided_slice_8/stack:output:01while/gru_cell_1/strided_slice_8/stack_1:output:01while/gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/gru_cell_1/MatMul_5MatMulwhile/gru_cell_1/mul:z:0)while/gru_cell_1/strided_slice_8:output:0*
T0*
_output_shapes
:	
while/gru_cell_1/add_2AddV2#while/gru_cell_1/BiasAdd_2:output:0#while/gru_cell_1/MatMul_5:product:0*
T0*
_output_shapes
:	c
while/gru_cell_1/TanhTanhwhile/gru_cell_1/add_2:z:0*
T0*
_output_shapes
:	z
while/gru_cell_1/mul_1Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*
_output_shapes
:	[
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*
_output_shapes
:	|
while/gru_cell_1/mul_2Mulwhile/gru_cell_1/sub:z:0while/gru_cell_1/Tanh:y:0*
T0*
_output_shapes
:	
while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_1:z:0while/gru_cell_1/mul_2:z:0*
T0*
_output_shapes
:	Ă
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:éčŇM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éčŇo
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0^while/NoOp*
T0*
_output_shapes
:	

while/NoOpNoOp ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6"^while/gru_cell_1/ReadVariableOp_7"^while/gru_cell_1/ReadVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 "Z
*while_gru_cell_1_readvariableop_3_resource,while_gru_cell_1_readvariableop_3_resource_0"Z
*while_gru_cell_1_readvariableop_6_resource,while_gru_cell_1_readvariableop_6_resource_0"V
(while_gru_cell_1_readvariableop_resource*while_gru_cell_1_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*0
_input_shapes
: : : : :	: : : : : 2B
while/gru_cell_1/ReadVariableOpwhile/gru_cell_1/ReadVariableOp2F
!while/gru_cell_1/ReadVariableOp_1!while/gru_cell_1/ReadVariableOp_12F
!while/gru_cell_1/ReadVariableOp_2!while/gru_cell_1/ReadVariableOp_22F
!while/gru_cell_1/ReadVariableOp_3!while/gru_cell_1/ReadVariableOp_32F
!while/gru_cell_1/ReadVariableOp_4!while/gru_cell_1/ReadVariableOp_42F
!while/gru_cell_1/ReadVariableOp_5!while/gru_cell_1/ReadVariableOp_52F
!while/gru_cell_1/ReadVariableOp_6!while/gru_cell_1/ReadVariableOp_62F
!while/gru_cell_1/ReadVariableOp_7!while/gru_cell_1/ReadVariableOp_72F
!while/gru_cell_1/ReadVariableOp_8!while/gru_cell_1/ReadVariableOp_8: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: 
ś
Ő
&__inference_gru_1_layer_call_fn_291210
inputs_0
unknown:	
	unknown_0:

	unknown_1:	
	unknown_2:

identity˘StatefulPartitionedCallů
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_289587t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/0
ćN

F__inference_gru_cell_1_layer_call_and_return_conditional_losses_289432

inputs
states:	+
readvariableop_resource:
(
readvariableop_3_resource:	-
readvariableop_6_resource:

identity

identity_1˘MatMul_3/ReadVariableOp˘MatMul_4/ReadVariableOp˘ReadVariableOp˘ReadVariableOp_1˘ReadVariableOp_2˘ReadVariableOp_3˘ReadVariableOp_4˘ReadVariableOp_5˘ReadVariableOp_6˘ReadVariableOp_7˘ReadVariableOp_8˘mul/ReadVariableOp˘mul_1/ReadVariableOph
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      í
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskZ
MatMulMatMulinputsstrided_slice:output:0*
T0*
_output_shapes
:	j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask^
MatMul_1MatMulinputsstrided_slice_1:output:0*
T0*
_output_shapes
:	j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask^
MatMul_2MatMulinputsstrided_slice_2:output:0*
T0*
_output_shapes
:	g
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes	
:*
dtype0_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: b
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*

begin_maskh
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*
_output_shapes
:	g
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes	
:*
dtype0`
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Đ
strided_slice_4StridedSliceReadVariableOp_4:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:l
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*
_output_shapes
:	g
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes	
:*
dtype0`
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ŕ
strided_slice_5StridedSliceReadVariableOp_5:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*
end_maskl
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*
_output_shapes
:	l
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource* 
_output_shapes
:
*
dtype0f
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_6StridedSliceReadVariableOp_6:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask_
MatMul_3/ReadVariableOpReadVariableOpstates*
_output_shapes
:	*
dtype0w
MatMul_3MatMulMatMul_3/ReadVariableOp:value:0strided_slice_6:output:0*
T0*
_output_shapes
:	l
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource* 
_output_shapes
:
*
dtype0f
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_7StridedSliceReadVariableOp_7:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask_
MatMul_4/ReadVariableOpReadVariableOpstates*
_output_shapes
:	*
dtype0w
MatMul_4MatMulMatMul_4/ReadVariableOp:value:0strided_slice_7:output:0*
T0*
_output_shapes
:	\
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*
_output_shapes
:	E
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:	`
add_1AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*
_output_shapes
:	I
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:	Z
mul/ReadVariableOpReadVariableOpstates*
_output_shapes
:	*
dtype0_
mulMulSigmoid_1:y:0mul/ReadVariableOp:value:0*
T0*
_output_shapes
:	l
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource* 
_output_shapes
:
*
dtype0f
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_8StridedSliceReadVariableOp_8:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask_
MatMul_5MatMulmul:z:0strided_slice_8:output:0*
T0*
_output_shapes
:	`
add_2AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*
_output_shapes
:	A
TanhTanh	add_2:z:0*
T0*
_output_shapes
:	\
mul_1/ReadVariableOpReadVariableOpstates*
_output_shapes
:	*
dtype0a
mul_1MulSigmoid:y:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Q
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:	I
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:	N
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:	P
IdentityIdentity	add_3:z:0^NoOp*
T0*
_output_shapes
:	R

Identity_1Identity	add_3:z:0^NoOp*
T0*
_output_shapes
:	Ď
NoOpNoOp^MatMul_3/ReadVariableOp^MatMul_4/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8^mul/ReadVariableOp^mul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:	: : : : 22
MatMul_3/ReadVariableOpMatMul_3/ReadVariableOp22
MatMul_4/ReadVariableOpMatMul_4/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82(
mul/ReadVariableOpmul/ReadVariableOp2,
mul_1/ReadVariableOpmul_1/ReadVariableOp:G C

_output_shapes
:	
 
_user_specified_nameinputs:&"
 
_user_specified_namestates
¨	
Ľ
G__inference_embedding_1_layer_call_and_return_conditional_losses_291197

inputs*
embedding_lookup_291191:	V
identity˘embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ź
embedding_lookupResourceGatherembedding_lookup_291191Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/291191*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0Ł
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/291191*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙x
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:˙˙˙˙˙˙˙˙˙: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ó	
Ř
+__inference_gru_cell_1_layer_call_fn_292190

inputs
states_0
unknown:

	unknown_0:	
	unknown_1:

identity

identity_1˘StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:	:	*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_289544g
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
:	i

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
:	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:	:	: : : 22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	
 
_user_specified_nameinputs:IE

_output_shapes
:	
"
_user_specified_name
states/0
ňM

F__inference_gru_cell_1_layer_call_and_return_conditional_losses_292544

inputs
states_0+
readvariableop_resource:
(
readvariableop_3_resource:	-
readvariableop_6_resource:

identity

identity_1˘MatMul_3/ReadVariableOp˘MatMul_4/ReadVariableOp˘ReadVariableOp˘ReadVariableOp_1˘ReadVariableOp_2˘ReadVariableOp_3˘ReadVariableOp_4˘ReadVariableOp_5˘ReadVariableOp_6˘ReadVariableOp_7˘ReadVariableOp_8˘mul/ReadVariableOp˘mul_1/ReadVariableOph
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      í
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskZ
MatMulMatMulinputsstrided_slice:output:0*
T0*
_output_shapes
:	j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask^
MatMul_1MatMulinputsstrided_slice_1:output:0*
T0*
_output_shapes
:	j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask^
MatMul_2MatMulinputsstrided_slice_2:output:0*
T0*
_output_shapes
:	g
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes	
:*
dtype0_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: b
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*

begin_maskh
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*
_output_shapes
:	g
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes	
:*
dtype0`
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Đ
strided_slice_4StridedSliceReadVariableOp_4:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:l
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*
_output_shapes
:	g
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes	
:*
dtype0`
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ŕ
strided_slice_5StridedSliceReadVariableOp_5:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*
end_maskl
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*
_output_shapes
:	l
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource* 
_output_shapes
:
*
dtype0f
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_6StridedSliceReadVariableOp_6:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskZ
MatMul_3/ReadVariableOpReadVariableOpstates_0*
_output_shapes
:*
dtype0w
MatMul_3BatchMatMulV2MatMul_3/ReadVariableOp:value:0strided_slice_6:output:0*
T0*
_output_shapes
:l
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource* 
_output_shapes
:
*
dtype0f
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_7StridedSliceReadVariableOp_7:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskZ
MatMul_4/ReadVariableOpReadVariableOpstates_0*
_output_shapes
:*
dtype0w
MatMul_4BatchMatMulV2MatMul_4/ReadVariableOp:value:0strided_slice_7:output:0*
T0*
_output_shapes
:T
addAddV2BiasAdd:output:0MatMul_3:output:0*
T0*
_output_shapes
:>
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:X
add_1AddV2BiasAdd_1:output:0MatMul_4:output:0*
T0*
_output_shapes
:B
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:U
mul/ReadVariableOpReadVariableOpstates_0*
_output_shapes
:*
dtype0X
mulMulSigmoid_1:y:0mul/ReadVariableOp:value:0*
T0*
_output_shapes
:l
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource* 
_output_shapes
:
*
dtype0f
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_8StridedSliceReadVariableOp_8:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask_
MatMul_5BatchMatMulV2mul:z:0strided_slice_8:output:0*
T0*
_output_shapes
:X
add_2AddV2BiasAdd_2:output:0MatMul_5:output:0*
T0*
_output_shapes
::
TanhTanh	add_2:z:0*
T0*
_output_shapes
:W
mul_1/ReadVariableOpReadVariableOpstates_0*
_output_shapes
:*
dtype0Z
mul_1MulSigmoid:y:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?J
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:B
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:G
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:I
IdentityIdentity	add_3:z:0^NoOp*
T0*
_output_shapes
:K

Identity_1Identity	add_3:z:0^NoOp*
T0*
_output_shapes
:Ď
NoOpNoOp^MatMul_3/ReadVariableOp^MatMul_4/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8^mul/ReadVariableOp^mul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:	:	: : : 22
MatMul_3/ReadVariableOpMatMul_3/ReadVariableOp22
MatMul_4/ReadVariableOpMatMul_4/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82(
mul/ReadVariableOpmul/ReadVariableOp2,
mul_1/ReadVariableOpmul_1/ReadVariableOp:G C

_output_shapes
:	
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0
Ľ	
ą
$__inference_signature_wrapper_291180
embedding_1_input
unknown:	V
	unknown_0:

	unknown_1:	
	unknown_2:

	unknown_3:	
	unknown_4:	V
	unknown_5:V
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallembedding_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙V*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_289331s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙V`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:˙˙˙˙˙˙˙˙˙: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameembedding_1_input
É

A__inference_gru_1_layer_call_and_return_conditional_losses_292137

inputs6
"gru_cell_1_readvariableop_resource:
3
$gru_cell_1_readvariableop_3_resource:	8
$gru_cell_1_readvariableop_6_resource:
>
+gru_cell_1_matmul_3_readvariableop_resource:	
identity˘AssignVariableOp˘ReadVariableOp˘"gru_cell_1/MatMul_3/ReadVariableOp˘"gru_cell_1/MatMul_4/ReadVariableOp˘gru_cell_1/ReadVariableOp˘gru_cell_1/ReadVariableOp_1˘gru_cell_1/ReadVariableOp_2˘gru_cell_1/ReadVariableOp_3˘gru_cell_1/ReadVariableOp_4˘gru_cell_1/ReadVariableOp_5˘gru_cell_1/ReadVariableOp_6˘gru_cell_1/ReadVariableOp_7˘gru_cell_1/ReadVariableOp_8˘gru_cell_1/mul/ReadVariableOp˘gru_cell_1/mul_1/ReadVariableOp˘whilec
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙˛
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:á
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_mask~
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype0o
gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        q
 gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       q
 gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¤
gru_cell_1/strided_sliceStridedSlice!gru_cell_1/ReadVariableOp:value:0'gru_cell_1/strided_slice/stack:output:0)gru_cell_1/strided_slice/stack_1:output:0)gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_cell_1/MatMulMatMulstrided_slice_1:output:0!gru_cell_1/strided_slice:output:0*
T0*
_output_shapes
:	
gru_cell_1/ReadVariableOp_1ReadVariableOp"gru_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype0q
 gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
gru_cell_1/strided_slice_1StridedSlice#gru_cell_1/ReadVariableOp_1:value:0)gru_cell_1/strided_slice_1/stack:output:0+gru_cell_1/strided_slice_1/stack_1:output:0+gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_cell_1/MatMul_1MatMulstrided_slice_1:output:0#gru_cell_1/strided_slice_1:output:0*
T0*
_output_shapes
:	
gru_cell_1/ReadVariableOp_2ReadVariableOp"gru_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype0q
 gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        s
"gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
gru_cell_1/strided_slice_2StridedSlice#gru_cell_1/ReadVariableOp_2:value:0)gru_cell_1/strided_slice_2/stack:output:0+gru_cell_1/strided_slice_2/stack_1:output:0+gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_cell_1/MatMul_2MatMulstrided_slice_1:output:0#gru_cell_1/strided_slice_2:output:0*
T0*
_output_shapes
:	}
gru_cell_1/ReadVariableOp_3ReadVariableOp$gru_cell_1_readvariableop_3_resource*
_output_shapes	
:*
dtype0j
 gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: m
"gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_cell_1/strided_slice_3StridedSlice#gru_cell_1/ReadVariableOp_3:value:0)gru_cell_1/strided_slice_3/stack:output:0+gru_cell_1/strided_slice_3/stack_1:output:0+gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*

begin_mask
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0#gru_cell_1/strided_slice_3:output:0*
T0*
_output_shapes
:	}
gru_cell_1/ReadVariableOp_4ReadVariableOp$gru_cell_1_readvariableop_3_resource*
_output_shapes	
:*
dtype0k
 gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:m
"gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_cell_1/strided_slice_4StridedSlice#gru_cell_1/ReadVariableOp_4:value:0)gru_cell_1/strided_slice_4/stack:output:0+gru_cell_1/strided_slice_4/stack_1:output:0+gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0#gru_cell_1/strided_slice_4:output:0*
T0*
_output_shapes
:	}
gru_cell_1/ReadVariableOp_5ReadVariableOp$gru_cell_1_readvariableop_3_resource*
_output_shapes	
:*
dtype0k
 gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:l
"gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: l
"gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_cell_1/strided_slice_5StridedSlice#gru_cell_1/ReadVariableOp_5:value:0)gru_cell_1/strided_slice_5/stack:output:0+gru_cell_1/strided_slice_5/stack_1:output:0+gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*
end_mask
gru_cell_1/BiasAdd_2BiasAddgru_cell_1/MatMul_2:product:0#gru_cell_1/strided_slice_5:output:0*
T0*
_output_shapes
:	
gru_cell_1/ReadVariableOp_6ReadVariableOp$gru_cell_1_readvariableop_6_resource* 
_output_shapes
:
*
dtype0q
 gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
gru_cell_1/strided_slice_6StridedSlice#gru_cell_1/ReadVariableOp_6:value:0)gru_cell_1/strided_slice_6/stack:output:0+gru_cell_1/strided_slice_6/stack_1:output:0+gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
"gru_cell_1/MatMul_3/ReadVariableOpReadVariableOp+gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_1/MatMul_3MatMul*gru_cell_1/MatMul_3/ReadVariableOp:value:0#gru_cell_1/strided_slice_6:output:0*
T0*
_output_shapes
:	
gru_cell_1/ReadVariableOp_7ReadVariableOp$gru_cell_1_readvariableop_6_resource* 
_output_shapes
:
*
dtype0q
 gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
gru_cell_1/strided_slice_7StridedSlice#gru_cell_1/ReadVariableOp_7:value:0)gru_cell_1/strided_slice_7/stack:output:0+gru_cell_1/strided_slice_7/stack_1:output:0+gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
"gru_cell_1/MatMul_4/ReadVariableOpReadVariableOp+gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_1/MatMul_4MatMul*gru_cell_1/MatMul_4/ReadVariableOp:value:0#gru_cell_1/strided_slice_7:output:0*
T0*
_output_shapes
:	}
gru_cell_1/addAddV2gru_cell_1/BiasAdd:output:0gru_cell_1/MatMul_3:product:0*
T0*
_output_shapes
:	[
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*
_output_shapes
:	
gru_cell_1/add_1AddV2gru_cell_1/BiasAdd_1:output:0gru_cell_1/MatMul_4:product:0*
T0*
_output_shapes
:	_
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*
_output_shapes
:	
gru_cell_1/mul/ReadVariableOpReadVariableOp+gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0%gru_cell_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:	
gru_cell_1/ReadVariableOp_8ReadVariableOp$gru_cell_1_readvariableop_6_resource* 
_output_shapes
:
*
dtype0q
 gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        s
"gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
gru_cell_1/strided_slice_8StridedSlice#gru_cell_1/ReadVariableOp_8:value:0)gru_cell_1/strided_slice_8/stack:output:0+gru_cell_1/strided_slice_8/stack_1:output:0+gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_cell_1/MatMul_5MatMulgru_cell_1/mul:z:0#gru_cell_1/strided_slice_8:output:0*
T0*
_output_shapes
:	
gru_cell_1/add_2AddV2gru_cell_1/BiasAdd_2:output:0gru_cell_1/MatMul_5:product:0*
T0*
_output_shapes
:	W
gru_cell_1/TanhTanhgru_cell_1/add_2:z:0*
T0*
_output_shapes
:	
gru_cell_1/mul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0'gru_cell_1/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	U
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?r
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*
_output_shapes
:	j
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Tanh:y:0*
T0*
_output_shapes
:	o
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*
_output_shapes
:	n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ś
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : {
ReadVariableOpReadVariableOp+gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ľ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0ReadVariableOp:value:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_1_readvariableop_resource$gru_cell_1_readvariableop_3_resource$gru_cell_1_readvariableop_6_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_292012*
condR
while_cond_292011*0
output_shapes
: : : : :	: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ă
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:˙
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    Ą
AssignVariableOpAssignVariableOp+gru_cell_1_matmul_3_readvariableop_resourcewhile:output:4^ReadVariableOp#^gru_cell_1/MatMul_3/ReadVariableOp#^gru_cell_1/MatMul_4/ReadVariableOp^gru_cell_1/mul/ReadVariableOp ^gru_cell_1/mul_1/ReadVariableOp*
_output_shapes
 *
dtype0c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp^AssignVariableOp^ReadVariableOp#^gru_cell_1/MatMul_3/ReadVariableOp#^gru_cell_1/MatMul_4/ReadVariableOp^gru_cell_1/ReadVariableOp^gru_cell_1/ReadVariableOp_1^gru_cell_1/ReadVariableOp_2^gru_cell_1/ReadVariableOp_3^gru_cell_1/ReadVariableOp_4^gru_cell_1/ReadVariableOp_5^gru_cell_1/ReadVariableOp_6^gru_cell_1/ReadVariableOp_7^gru_cell_1/ReadVariableOp_8^gru_cell_1/mul/ReadVariableOp ^gru_cell_1/mul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙: : : : 2$
AssignVariableOpAssignVariableOp2 
ReadVariableOpReadVariableOp2H
"gru_cell_1/MatMul_3/ReadVariableOp"gru_cell_1/MatMul_3/ReadVariableOp2H
"gru_cell_1/MatMul_4/ReadVariableOp"gru_cell_1/MatMul_4/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2:
gru_cell_1/ReadVariableOp_1gru_cell_1/ReadVariableOp_12:
gru_cell_1/ReadVariableOp_2gru_cell_1/ReadVariableOp_22:
gru_cell_1/ReadVariableOp_3gru_cell_1/ReadVariableOp_32:
gru_cell_1/ReadVariableOp_4gru_cell_1/ReadVariableOp_42:
gru_cell_1/ReadVariableOp_5gru_cell_1/ReadVariableOp_52:
gru_cell_1/ReadVariableOp_6gru_cell_1/ReadVariableOp_62:
gru_cell_1/ReadVariableOp_7gru_cell_1/ReadVariableOp_72:
gru_cell_1/ReadVariableOp_8gru_cell_1/ReadVariableOp_82>
gru_cell_1/mul/ReadVariableOpgru_cell_1/mul/ReadVariableOp2B
gru_cell_1/mul_1/ReadVariableOpgru_cell_1/mul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ć
¨
while_cond_289447
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice4
0while_while_cond_289447___redundant_placeholder04
0while_while_cond_289447___redundant_placeholder14
0while_while_cond_289447___redundant_placeholder24
0while_while_cond_289447___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :	: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
:
Ď
ű
C__inference_dense_1_layer_call_and_return_conditional_losses_292176

inputs4
!tensordot_readvariableop_resource:	V-
biasadd_readvariableop_resource:V
identity˘BiasAdd/ReadVariableOp˘Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	V*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ť
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ż
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙V[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:VY
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙Vr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:V*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙Vc
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙Vz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
/
Ç
A__inference_gru_1_layer_call_and_return_conditional_losses_289587

inputs$
gru_cell_1_289433:	%
gru_cell_1_289435:
 
gru_cell_1_289437:	%
gru_cell_1_289439:

identity˘AssignVariableOp˘ReadVariableOp˘"gru_cell_1/StatefulPartitionedCall˘whilec
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙˛
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:á
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_maskť
"gru_cell_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0gru_cell_1_289433gru_cell_1_289435gru_cell_1_289437gru_cell_1_289439*
Tin	
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:	:	*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_289432n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ś
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : a
ReadVariableOpReadVariableOpgru_cell_1_289433*
_output_shapes
:	*
dtype0c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : î
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0ReadVariableOp:value:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_1_289435gru_cell_1_289437gru_cell_1_289439*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_289448*
condR
while_cond_289447*0
output_shapes
: : : : :	: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ă
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:˙
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *     
AssignVariableOpAssignVariableOpgru_cell_1_289433while:output:4^ReadVariableOp#^gru_cell_1/StatefulPartitionedCall*
_output_shapes
 *
dtype0c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp^AssignVariableOp^ReadVariableOp#^gru_cell_1/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙: : : : 2$
AssignVariableOpAssignVariableOp2 
ReadVariableOpReadVariableOp2H
"gru_cell_1/StatefulPartitionedCall"gru_cell_1/StatefulPartitionedCall2
whilewhile:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
´	
Ż
-__inference_sequential_1_layer_call_fn_290651

inputs
unknown:	V
	unknown_0:

	unknown_1:	
	unknown_2:

	unknown_3:	
	unknown_4:	V
	unknown_5:V
identity˘StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙V*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_290535s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙V`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:˙˙˙˙˙˙˙˙˙: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ó	
Ř
+__inference_gru_cell_1_layer_call_fn_292219

inputs
states_0
unknown:

	unknown_0:	
	unknown_1:

identity

identity_1˘StatefulPartitionedCallű
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:	:	*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_289432g
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
:	i

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
:	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:	:	: : : 22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0
Ľ
Ţ
__inference__traced_save_292588
file_prefix5
1savev2_embedding_1_embeddings_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop6
2savev2_gru_1_gru_cell_1_kernel_read_readvariableop@
<savev2_gru_1_gru_cell_1_recurrent_kernel_read_readvariableop4
0savev2_gru_1_gru_cell_1_bias_read_readvariableop-
)savev2_gru_1_variable_read_readvariableop
savev2_const

identity_1˘MergeV2Checkpointsw
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
: đ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-1/keras_api/states/0/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH}
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_embedding_1_embeddings_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop2savev2_gru_1_gru_cell_1_kernel_read_readvariableop<savev2_gru_1_gru_cell_1_recurrent_kernel_read_readvariableop0savev2_gru_1_gru_cell_1_bias_read_readvariableop)savev2_gru_1_variable_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes

2
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

identity_1Identity_1:output:0*]
_input_shapesL
J: :	V:	V:V:
:
::	: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	V:%!

_output_shapes
:	V: 

_output_shapes
:V:&"
 
_output_shapes
:
:&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	:

_output_shapes
: 
É

A__inference_gru_1_layer_call_and_return_conditional_losses_290163

inputs6
"gru_cell_1_readvariableop_resource:
3
$gru_cell_1_readvariableop_3_resource:	8
$gru_cell_1_readvariableop_6_resource:
>
+gru_cell_1_matmul_3_readvariableop_resource:	
identity˘AssignVariableOp˘ReadVariableOp˘"gru_cell_1/MatMul_3/ReadVariableOp˘"gru_cell_1/MatMul_4/ReadVariableOp˘gru_cell_1/ReadVariableOp˘gru_cell_1/ReadVariableOp_1˘gru_cell_1/ReadVariableOp_2˘gru_cell_1/ReadVariableOp_3˘gru_cell_1/ReadVariableOp_4˘gru_cell_1/ReadVariableOp_5˘gru_cell_1/ReadVariableOp_6˘gru_cell_1/ReadVariableOp_7˘gru_cell_1/ReadVariableOp_8˘gru_cell_1/mul/ReadVariableOp˘gru_cell_1/mul_1/ReadVariableOp˘whilec
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙˛
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:á
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_mask~
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype0o
gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        q
 gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       q
 gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¤
gru_cell_1/strided_sliceStridedSlice!gru_cell_1/ReadVariableOp:value:0'gru_cell_1/strided_slice/stack:output:0)gru_cell_1/strided_slice/stack_1:output:0)gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_cell_1/MatMulMatMulstrided_slice_1:output:0!gru_cell_1/strided_slice:output:0*
T0*
_output_shapes
:	
gru_cell_1/ReadVariableOp_1ReadVariableOp"gru_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype0q
 gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
gru_cell_1/strided_slice_1StridedSlice#gru_cell_1/ReadVariableOp_1:value:0)gru_cell_1/strided_slice_1/stack:output:0+gru_cell_1/strided_slice_1/stack_1:output:0+gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_cell_1/MatMul_1MatMulstrided_slice_1:output:0#gru_cell_1/strided_slice_1:output:0*
T0*
_output_shapes
:	
gru_cell_1/ReadVariableOp_2ReadVariableOp"gru_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype0q
 gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        s
"gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
gru_cell_1/strided_slice_2StridedSlice#gru_cell_1/ReadVariableOp_2:value:0)gru_cell_1/strided_slice_2/stack:output:0+gru_cell_1/strided_slice_2/stack_1:output:0+gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_cell_1/MatMul_2MatMulstrided_slice_1:output:0#gru_cell_1/strided_slice_2:output:0*
T0*
_output_shapes
:	}
gru_cell_1/ReadVariableOp_3ReadVariableOp$gru_cell_1_readvariableop_3_resource*
_output_shapes	
:*
dtype0j
 gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: m
"gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_cell_1/strided_slice_3StridedSlice#gru_cell_1/ReadVariableOp_3:value:0)gru_cell_1/strided_slice_3/stack:output:0+gru_cell_1/strided_slice_3/stack_1:output:0+gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*

begin_mask
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0#gru_cell_1/strided_slice_3:output:0*
T0*
_output_shapes
:	}
gru_cell_1/ReadVariableOp_4ReadVariableOp$gru_cell_1_readvariableop_3_resource*
_output_shapes	
:*
dtype0k
 gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:m
"gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_cell_1/strided_slice_4StridedSlice#gru_cell_1/ReadVariableOp_4:value:0)gru_cell_1/strided_slice_4/stack:output:0+gru_cell_1/strided_slice_4/stack_1:output:0+gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0#gru_cell_1/strided_slice_4:output:0*
T0*
_output_shapes
:	}
gru_cell_1/ReadVariableOp_5ReadVariableOp$gru_cell_1_readvariableop_3_resource*
_output_shapes	
:*
dtype0k
 gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:l
"gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: l
"gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_cell_1/strided_slice_5StridedSlice#gru_cell_1/ReadVariableOp_5:value:0)gru_cell_1/strided_slice_5/stack:output:0+gru_cell_1/strided_slice_5/stack_1:output:0+gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*
end_mask
gru_cell_1/BiasAdd_2BiasAddgru_cell_1/MatMul_2:product:0#gru_cell_1/strided_slice_5:output:0*
T0*
_output_shapes
:	
gru_cell_1/ReadVariableOp_6ReadVariableOp$gru_cell_1_readvariableop_6_resource* 
_output_shapes
:
*
dtype0q
 gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
gru_cell_1/strided_slice_6StridedSlice#gru_cell_1/ReadVariableOp_6:value:0)gru_cell_1/strided_slice_6/stack:output:0+gru_cell_1/strided_slice_6/stack_1:output:0+gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
"gru_cell_1/MatMul_3/ReadVariableOpReadVariableOp+gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_1/MatMul_3MatMul*gru_cell_1/MatMul_3/ReadVariableOp:value:0#gru_cell_1/strided_slice_6:output:0*
T0*
_output_shapes
:	
gru_cell_1/ReadVariableOp_7ReadVariableOp$gru_cell_1_readvariableop_6_resource* 
_output_shapes
:
*
dtype0q
 gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
gru_cell_1/strided_slice_7StridedSlice#gru_cell_1/ReadVariableOp_7:value:0)gru_cell_1/strided_slice_7/stack:output:0+gru_cell_1/strided_slice_7/stack_1:output:0+gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
"gru_cell_1/MatMul_4/ReadVariableOpReadVariableOp+gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_1/MatMul_4MatMul*gru_cell_1/MatMul_4/ReadVariableOp:value:0#gru_cell_1/strided_slice_7:output:0*
T0*
_output_shapes
:	}
gru_cell_1/addAddV2gru_cell_1/BiasAdd:output:0gru_cell_1/MatMul_3:product:0*
T0*
_output_shapes
:	[
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*
_output_shapes
:	
gru_cell_1/add_1AddV2gru_cell_1/BiasAdd_1:output:0gru_cell_1/MatMul_4:product:0*
T0*
_output_shapes
:	_
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*
_output_shapes
:	
gru_cell_1/mul/ReadVariableOpReadVariableOp+gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0%gru_cell_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:	
gru_cell_1/ReadVariableOp_8ReadVariableOp$gru_cell_1_readvariableop_6_resource* 
_output_shapes
:
*
dtype0q
 gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        s
"gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
gru_cell_1/strided_slice_8StridedSlice#gru_cell_1/ReadVariableOp_8:value:0)gru_cell_1/strided_slice_8/stack:output:0+gru_cell_1/strided_slice_8/stack_1:output:0+gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_cell_1/MatMul_5MatMulgru_cell_1/mul:z:0#gru_cell_1/strided_slice_8:output:0*
T0*
_output_shapes
:	
gru_cell_1/add_2AddV2gru_cell_1/BiasAdd_2:output:0gru_cell_1/MatMul_5:product:0*
T0*
_output_shapes
:	W
gru_cell_1/TanhTanhgru_cell_1/add_2:z:0*
T0*
_output_shapes
:	
gru_cell_1/mul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0'gru_cell_1/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	U
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?r
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*
_output_shapes
:	j
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Tanh:y:0*
T0*
_output_shapes
:	o
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*
_output_shapes
:	n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ś
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : {
ReadVariableOpReadVariableOp+gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ľ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0ReadVariableOp:value:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_1_readvariableop_resource$gru_cell_1_readvariableop_3_resource$gru_cell_1_readvariableop_6_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_290038*
condR
while_cond_290037*0
output_shapes
: : : : :	: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ă
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:˙
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    Ą
AssignVariableOpAssignVariableOp+gru_cell_1_matmul_3_readvariableop_resourcewhile:output:4^ReadVariableOp#^gru_cell_1/MatMul_3/ReadVariableOp#^gru_cell_1/MatMul_4/ReadVariableOp^gru_cell_1/mul/ReadVariableOp ^gru_cell_1/mul_1/ReadVariableOp*
_output_shapes
 *
dtype0c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp^AssignVariableOp^ReadVariableOp#^gru_cell_1/MatMul_3/ReadVariableOp#^gru_cell_1/MatMul_4/ReadVariableOp^gru_cell_1/ReadVariableOp^gru_cell_1/ReadVariableOp_1^gru_cell_1/ReadVariableOp_2^gru_cell_1/ReadVariableOp_3^gru_cell_1/ReadVariableOp_4^gru_cell_1/ReadVariableOp_5^gru_cell_1/ReadVariableOp_6^gru_cell_1/ReadVariableOp_7^gru_cell_1/ReadVariableOp_8^gru_cell_1/mul/ReadVariableOp ^gru_cell_1/mul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙: : : : 2$
AssignVariableOpAssignVariableOp2 
ReadVariableOpReadVariableOp2H
"gru_cell_1/MatMul_3/ReadVariableOp"gru_cell_1/MatMul_3/ReadVariableOp2H
"gru_cell_1/MatMul_4/ReadVariableOp"gru_cell_1/MatMul_4/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2:
gru_cell_1/ReadVariableOp_1gru_cell_1/ReadVariableOp_12:
gru_cell_1/ReadVariableOp_2gru_cell_1/ReadVariableOp_22:
gru_cell_1/ReadVariableOp_3gru_cell_1/ReadVariableOp_32:
gru_cell_1/ReadVariableOp_4gru_cell_1/ReadVariableOp_42:
gru_cell_1/ReadVariableOp_5gru_cell_1/ReadVariableOp_52:
gru_cell_1/ReadVariableOp_6gru_cell_1/ReadVariableOp_62:
gru_cell_1/ReadVariableOp_7gru_cell_1/ReadVariableOp_72:
gru_cell_1/ReadVariableOp_8gru_cell_1/ReadVariableOp_82>
gru_cell_1/mul/ReadVariableOpgru_cell_1/mul/ReadVariableOp2B
gru_cell_1/mul_1/ReadVariableOpgru_cell_1/mul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ć
¨
while_cond_292011
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice4
0while_while_cond_292011___redundant_placeholder04
0while_while_cond_292011___redundant_placeholder14
0while_while_cond_292011___redundant_placeholder24
0while_while_cond_292011___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :	: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
:
°
Ó
&__inference_gru_1_layer_call_fn_291236

inputs
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
identity˘StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_290163t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ć
¨
while_cond_291789
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice4
0while_while_cond_291789___redundant_placeholder04
0while_while_cond_291789___redundant_placeholder14
0while_while_cond_291789___redundant_placeholder24
0while_while_cond_291789___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :	: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
:
ó	
Ř
+__inference_gru_cell_1_layer_call_fn_292204

inputs
states_0
unknown:

	unknown_0:	
	unknown_1:

identity

identity_1˘StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:	:	*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_289691g
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
:	i

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
:	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:	:	: : : 22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	
 
_user_specified_nameinputs:IE

_output_shapes
:	
"
_user_specified_name
states/0
Ć
¨
while_cond_291345
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice4
0while_while_cond_291345___redundant_placeholder04
0while_while_cond_291345___redundant_placeholder14
0while_while_cond_291345___redundant_placeholder24
0while_while_cond_291345___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :	: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
:
Ď

A__inference_gru_1_layer_call_and_return_conditional_losses_291693
inputs_06
"gru_cell_1_readvariableop_resource:
3
$gru_cell_1_readvariableop_3_resource:	8
$gru_cell_1_readvariableop_6_resource:
>
+gru_cell_1_matmul_3_readvariableop_resource:	
identity˘AssignVariableOp˘ReadVariableOp˘"gru_cell_1/MatMul_3/ReadVariableOp˘"gru_cell_1/MatMul_4/ReadVariableOp˘gru_cell_1/ReadVariableOp˘gru_cell_1/ReadVariableOp_1˘gru_cell_1/ReadVariableOp_2˘gru_cell_1/ReadVariableOp_3˘gru_cell_1/ReadVariableOp_4˘gru_cell_1/ReadVariableOp_5˘gru_cell_1/ReadVariableOp_6˘gru_cell_1/ReadVariableOp_7˘gru_cell_1/ReadVariableOp_8˘gru_cell_1/mul/ReadVariableOp˘gru_cell_1/mul_1/ReadVariableOp˘whilec
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          p
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙˛
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:á
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_mask~
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype0o
gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        q
 gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       q
 gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¤
gru_cell_1/strided_sliceStridedSlice!gru_cell_1/ReadVariableOp:value:0'gru_cell_1/strided_slice/stack:output:0)gru_cell_1/strided_slice/stack_1:output:0)gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_cell_1/MatMulMatMulstrided_slice_1:output:0!gru_cell_1/strided_slice:output:0*
T0*
_output_shapes
:	
gru_cell_1/ReadVariableOp_1ReadVariableOp"gru_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype0q
 gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
gru_cell_1/strided_slice_1StridedSlice#gru_cell_1/ReadVariableOp_1:value:0)gru_cell_1/strided_slice_1/stack:output:0+gru_cell_1/strided_slice_1/stack_1:output:0+gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_cell_1/MatMul_1MatMulstrided_slice_1:output:0#gru_cell_1/strided_slice_1:output:0*
T0*
_output_shapes
:	
gru_cell_1/ReadVariableOp_2ReadVariableOp"gru_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype0q
 gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        s
"gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
gru_cell_1/strided_slice_2StridedSlice#gru_cell_1/ReadVariableOp_2:value:0)gru_cell_1/strided_slice_2/stack:output:0+gru_cell_1/strided_slice_2/stack_1:output:0+gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_cell_1/MatMul_2MatMulstrided_slice_1:output:0#gru_cell_1/strided_slice_2:output:0*
T0*
_output_shapes
:	}
gru_cell_1/ReadVariableOp_3ReadVariableOp$gru_cell_1_readvariableop_3_resource*
_output_shapes	
:*
dtype0j
 gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: m
"gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_cell_1/strided_slice_3StridedSlice#gru_cell_1/ReadVariableOp_3:value:0)gru_cell_1/strided_slice_3/stack:output:0+gru_cell_1/strided_slice_3/stack_1:output:0+gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*

begin_mask
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0#gru_cell_1/strided_slice_3:output:0*
T0*
_output_shapes
:	}
gru_cell_1/ReadVariableOp_4ReadVariableOp$gru_cell_1_readvariableop_3_resource*
_output_shapes	
:*
dtype0k
 gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:m
"gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_cell_1/strided_slice_4StridedSlice#gru_cell_1/ReadVariableOp_4:value:0)gru_cell_1/strided_slice_4/stack:output:0+gru_cell_1/strided_slice_4/stack_1:output:0+gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0#gru_cell_1/strided_slice_4:output:0*
T0*
_output_shapes
:	}
gru_cell_1/ReadVariableOp_5ReadVariableOp$gru_cell_1_readvariableop_3_resource*
_output_shapes	
:*
dtype0k
 gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:l
"gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: l
"gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_cell_1/strided_slice_5StridedSlice#gru_cell_1/ReadVariableOp_5:value:0)gru_cell_1/strided_slice_5/stack:output:0+gru_cell_1/strided_slice_5/stack_1:output:0+gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*
end_mask
gru_cell_1/BiasAdd_2BiasAddgru_cell_1/MatMul_2:product:0#gru_cell_1/strided_slice_5:output:0*
T0*
_output_shapes
:	
gru_cell_1/ReadVariableOp_6ReadVariableOp$gru_cell_1_readvariableop_6_resource* 
_output_shapes
:
*
dtype0q
 gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
gru_cell_1/strided_slice_6StridedSlice#gru_cell_1/ReadVariableOp_6:value:0)gru_cell_1/strided_slice_6/stack:output:0+gru_cell_1/strided_slice_6/stack_1:output:0+gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
"gru_cell_1/MatMul_3/ReadVariableOpReadVariableOp+gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_1/MatMul_3MatMul*gru_cell_1/MatMul_3/ReadVariableOp:value:0#gru_cell_1/strided_slice_6:output:0*
T0*
_output_shapes
:	
gru_cell_1/ReadVariableOp_7ReadVariableOp$gru_cell_1_readvariableop_6_resource* 
_output_shapes
:
*
dtype0q
 gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
gru_cell_1/strided_slice_7StridedSlice#gru_cell_1/ReadVariableOp_7:value:0)gru_cell_1/strided_slice_7/stack:output:0+gru_cell_1/strided_slice_7/stack_1:output:0+gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
"gru_cell_1/MatMul_4/ReadVariableOpReadVariableOp+gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_1/MatMul_4MatMul*gru_cell_1/MatMul_4/ReadVariableOp:value:0#gru_cell_1/strided_slice_7:output:0*
T0*
_output_shapes
:	}
gru_cell_1/addAddV2gru_cell_1/BiasAdd:output:0gru_cell_1/MatMul_3:product:0*
T0*
_output_shapes
:	[
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*
_output_shapes
:	
gru_cell_1/add_1AddV2gru_cell_1/BiasAdd_1:output:0gru_cell_1/MatMul_4:product:0*
T0*
_output_shapes
:	_
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*
_output_shapes
:	
gru_cell_1/mul/ReadVariableOpReadVariableOp+gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0%gru_cell_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:	
gru_cell_1/ReadVariableOp_8ReadVariableOp$gru_cell_1_readvariableop_6_resource* 
_output_shapes
:
*
dtype0q
 gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        s
"gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
gru_cell_1/strided_slice_8StridedSlice#gru_cell_1/ReadVariableOp_8:value:0)gru_cell_1/strided_slice_8/stack:output:0+gru_cell_1/strided_slice_8/stack_1:output:0+gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_cell_1/MatMul_5MatMulgru_cell_1/mul:z:0#gru_cell_1/strided_slice_8:output:0*
T0*
_output_shapes
:	
gru_cell_1/add_2AddV2gru_cell_1/BiasAdd_2:output:0gru_cell_1/MatMul_5:product:0*
T0*
_output_shapes
:	W
gru_cell_1/TanhTanhgru_cell_1/add_2:z:0*
T0*
_output_shapes
:	
gru_cell_1/mul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0'gru_cell_1/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	U
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?r
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*
_output_shapes
:	j
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Tanh:y:0*
T0*
_output_shapes
:	o
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*
_output_shapes
:	n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ś
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : {
ReadVariableOpReadVariableOp+gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ľ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0ReadVariableOp:value:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_1_readvariableop_resource$gru_cell_1_readvariableop_3_resource$gru_cell_1_readvariableop_6_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_291568*
condR
while_cond_291567*0
output_shapes
: : : : :	: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ă
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:˙
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    Ą
AssignVariableOpAssignVariableOp+gru_cell_1_matmul_3_readvariableop_resourcewhile:output:4^ReadVariableOp#^gru_cell_1/MatMul_3/ReadVariableOp#^gru_cell_1/MatMul_4/ReadVariableOp^gru_cell_1/mul/ReadVariableOp ^gru_cell_1/mul_1/ReadVariableOp*
_output_shapes
 *
dtype0c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp^AssignVariableOp^ReadVariableOp#^gru_cell_1/MatMul_3/ReadVariableOp#^gru_cell_1/MatMul_4/ReadVariableOp^gru_cell_1/ReadVariableOp^gru_cell_1/ReadVariableOp_1^gru_cell_1/ReadVariableOp_2^gru_cell_1/ReadVariableOp_3^gru_cell_1/ReadVariableOp_4^gru_cell_1/ReadVariableOp_5^gru_cell_1/ReadVariableOp_6^gru_cell_1/ReadVariableOp_7^gru_cell_1/ReadVariableOp_8^gru_cell_1/mul/ReadVariableOp ^gru_cell_1/mul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙: : : : 2$
AssignVariableOpAssignVariableOp2 
ReadVariableOpReadVariableOp2H
"gru_cell_1/MatMul_3/ReadVariableOp"gru_cell_1/MatMul_3/ReadVariableOp2H
"gru_cell_1/MatMul_4/ReadVariableOp"gru_cell_1/MatMul_4/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2:
gru_cell_1/ReadVariableOp_1gru_cell_1/ReadVariableOp_12:
gru_cell_1/ReadVariableOp_2gru_cell_1/ReadVariableOp_22:
gru_cell_1/ReadVariableOp_3gru_cell_1/ReadVariableOp_32:
gru_cell_1/ReadVariableOp_4gru_cell_1/ReadVariableOp_42:
gru_cell_1/ReadVariableOp_5gru_cell_1/ReadVariableOp_52:
gru_cell_1/ReadVariableOp_6gru_cell_1/ReadVariableOp_62:
gru_cell_1/ReadVariableOp_7gru_cell_1/ReadVariableOp_72:
gru_cell_1/ReadVariableOp_8gru_cell_1/ReadVariableOp_82>
gru_cell_1/mul/ReadVariableOpgru_cell_1/mul/ReadVariableOp2B
gru_cell_1/mul_1/ReadVariableOpgru_cell_1/mul_1/ReadVariableOp2
whilewhile:V R
,
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/0
Ĺ
š
H__inference_sequential_1_layer_call_and_return_conditional_losses_290210

inputs%
embedding_1_289939:	V 
gru_1_290164:

gru_1_290166:	 
gru_1_290168:

gru_1_290170:	!
dense_1_290204:	V
dense_1_290206:V
identity˘dense_1/StatefulPartitionedCall˘#embedding_1/StatefulPartitionedCall˘gru_1/StatefulPartitionedCallî
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_1_289939*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_embedding_1_layer_call_and_return_conditional_losses_289938ą
gru_1/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0gru_1_290164gru_1_290166gru_1_290168gru_1_290170*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_290163
dense_1/StatefulPartitionedCallStatefulPartitionedCall&gru_1/StatefulPartitionedCall:output:0dense_1_290204dense_1_290206*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙V*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_290203{
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙VŽ
NoOpNoOp ^dense_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:˙˙˙˙˙˙˙˙˙: : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Č

$sequential_1_gru_1_while_cond_289179B
>sequential_1_gru_1_while_sequential_1_gru_1_while_loop_counterH
Dsequential_1_gru_1_while_sequential_1_gru_1_while_maximum_iterations(
$sequential_1_gru_1_while_placeholder*
&sequential_1_gru_1_while_placeholder_1*
&sequential_1_gru_1_while_placeholder_2B
>sequential_1_gru_1_while_less_sequential_1_gru_1_strided_sliceZ
Vsequential_1_gru_1_while_sequential_1_gru_1_while_cond_289179___redundant_placeholder0Z
Vsequential_1_gru_1_while_sequential_1_gru_1_while_cond_289179___redundant_placeholder1Z
Vsequential_1_gru_1_while_sequential_1_gru_1_while_cond_289179___redundant_placeholder2Z
Vsequential_1_gru_1_while_sequential_1_gru_1_while_cond_289179___redundant_placeholder3%
!sequential_1_gru_1_while_identity
Ź
sequential_1/gru_1/while/LessLess$sequential_1_gru_1_while_placeholder>sequential_1_gru_1_while_less_sequential_1_gru_1_strided_slice*
T0*
_output_shapes
: q
!sequential_1/gru_1/while/IdentityIdentity!sequential_1/gru_1/while/Less:z:0*
T0
*
_output_shapes
: "O
!sequential_1_gru_1_while_identity*sequential_1/gru_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :	: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
:
ţá
Đ
!__inference__wrapped_model_289331
embedding_1_inputC
0sequential_1_embedding_1_embedding_lookup_289081:	VI
5sequential_1_gru_1_gru_cell_1_readvariableop_resource:
F
7sequential_1_gru_1_gru_cell_1_readvariableop_3_resource:	K
7sequential_1_gru_1_gru_cell_1_readvariableop_6_resource:
Q
>sequential_1_gru_1_gru_cell_1_matmul_3_readvariableop_resource:	I
6sequential_1_dense_1_tensordot_readvariableop_resource:	VB
4sequential_1_dense_1_biasadd_readvariableop_resource:V
identity˘+sequential_1/dense_1/BiasAdd/ReadVariableOp˘-sequential_1/dense_1/Tensordot/ReadVariableOp˘)sequential_1/embedding_1/embedding_lookup˘#sequential_1/gru_1/AssignVariableOp˘!sequential_1/gru_1/ReadVariableOp˘5sequential_1/gru_1/gru_cell_1/MatMul_3/ReadVariableOp˘5sequential_1/gru_1/gru_cell_1/MatMul_4/ReadVariableOp˘,sequential_1/gru_1/gru_cell_1/ReadVariableOp˘.sequential_1/gru_1/gru_cell_1/ReadVariableOp_1˘.sequential_1/gru_1/gru_cell_1/ReadVariableOp_2˘.sequential_1/gru_1/gru_cell_1/ReadVariableOp_3˘.sequential_1/gru_1/gru_cell_1/ReadVariableOp_4˘.sequential_1/gru_1/gru_cell_1/ReadVariableOp_5˘.sequential_1/gru_1/gru_cell_1/ReadVariableOp_6˘.sequential_1/gru_1/gru_cell_1/ReadVariableOp_7˘.sequential_1/gru_1/gru_cell_1/ReadVariableOp_8˘0sequential_1/gru_1/gru_cell_1/mul/ReadVariableOp˘2sequential_1/gru_1/gru_cell_1/mul_1/ReadVariableOp˘sequential_1/gru_1/whiley
sequential_1/embedding_1/CastCastembedding_1_input*

DstT0*

SrcT0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
)sequential_1/embedding_1/embedding_lookupResourceGather0sequential_1_embedding_1_embedding_lookup_289081!sequential_1/embedding_1/Cast:y:0*
Tindices0*C
_class9
75loc:@sequential_1/embedding_1/embedding_lookup/289081*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0î
2sequential_1/embedding_1/embedding_lookup/IdentityIdentity2sequential_1/embedding_1/embedding_lookup:output:0*
T0*C
_class9
75loc:@sequential_1/embedding_1/embedding_lookup/289081*,
_output_shapes
:˙˙˙˙˙˙˙˙˙´
4sequential_1/embedding_1/embedding_lookup/Identity_1Identity;sequential_1/embedding_1/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙v
!sequential_1/gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ë
sequential_1/gru_1/transpose	Transpose=sequential_1/embedding_1/embedding_lookup/Identity_1:output:0*sequential_1/gru_1/transpose/perm:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙h
sequential_1/gru_1/ShapeShape sequential_1/gru_1/transpose:y:0*
T0*
_output_shapes
:p
&sequential_1/gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(sequential_1/gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(sequential_1/gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 sequential_1/gru_1/strided_sliceStridedSlice!sequential_1/gru_1/Shape:output:0/sequential_1/gru_1/strided_slice/stack:output:01sequential_1/gru_1/strided_slice/stack_1:output:01sequential_1/gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
.sequential_1/gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙ë
 sequential_1/gru_1/TensorArrayV2TensorListReserve7sequential_1/gru_1/TensorArrayV2/element_shape:output:0)sequential_1/gru_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
Hsequential_1/gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
:sequential_1/gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor sequential_1/gru_1/transpose:y:0Qsequential_1/gru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇr
(sequential_1/gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_1/gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_1/gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ŕ
"sequential_1/gru_1/strided_slice_1StridedSlice sequential_1/gru_1/transpose:y:01sequential_1/gru_1/strided_slice_1/stack:output:03sequential_1/gru_1/strided_slice_1/stack_1:output:03sequential_1/gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_mask¤
,sequential_1/gru_1/gru_cell_1/ReadVariableOpReadVariableOp5sequential_1_gru_1_gru_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
1sequential_1/gru_1/gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
3sequential_1/gru_1/gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
3sequential_1/gru_1/gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
+sequential_1/gru_1/gru_cell_1/strided_sliceStridedSlice4sequential_1/gru_1/gru_cell_1/ReadVariableOp:value:0:sequential_1/gru_1/gru_cell_1/strided_slice/stack:output:0<sequential_1/gru_1/gru_cell_1/strided_slice/stack_1:output:0<sequential_1/gru_1/gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskť
$sequential_1/gru_1/gru_cell_1/MatMulMatMul+sequential_1/gru_1/strided_slice_1:output:04sequential_1/gru_1/gru_cell_1/strided_slice:output:0*
T0*
_output_shapes
:	Ś
.sequential_1/gru_1/gru_cell_1/ReadVariableOp_1ReadVariableOp5sequential_1_gru_1_gru_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
3sequential_1/gru_1/gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
5sequential_1/gru_1/gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
5sequential_1/gru_1/gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
-sequential_1/gru_1/gru_cell_1/strided_slice_1StridedSlice6sequential_1/gru_1/gru_cell_1/ReadVariableOp_1:value:0<sequential_1/gru_1/gru_cell_1/strided_slice_1/stack:output:0>sequential_1/gru_1/gru_cell_1/strided_slice_1/stack_1:output:0>sequential_1/gru_1/gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskż
&sequential_1/gru_1/gru_cell_1/MatMul_1MatMul+sequential_1/gru_1/strided_slice_1:output:06sequential_1/gru_1/gru_cell_1/strided_slice_1:output:0*
T0*
_output_shapes
:	Ś
.sequential_1/gru_1/gru_cell_1/ReadVariableOp_2ReadVariableOp5sequential_1_gru_1_gru_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
3sequential_1/gru_1/gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
5sequential_1/gru_1/gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
5sequential_1/gru_1/gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
-sequential_1/gru_1/gru_cell_1/strided_slice_2StridedSlice6sequential_1/gru_1/gru_cell_1/ReadVariableOp_2:value:0<sequential_1/gru_1/gru_cell_1/strided_slice_2/stack:output:0>sequential_1/gru_1/gru_cell_1/strided_slice_2/stack_1:output:0>sequential_1/gru_1/gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskż
&sequential_1/gru_1/gru_cell_1/MatMul_2MatMul+sequential_1/gru_1/strided_slice_1:output:06sequential_1/gru_1/gru_cell_1/strided_slice_2:output:0*
T0*
_output_shapes
:	Ł
.sequential_1/gru_1/gru_cell_1/ReadVariableOp_3ReadVariableOp7sequential_1_gru_1_gru_cell_1_readvariableop_3_resource*
_output_shapes	
:*
dtype0}
3sequential_1/gru_1/gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5sequential_1/gru_1/gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5sequential_1/gru_1/gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ř
-sequential_1/gru_1/gru_cell_1/strided_slice_3StridedSlice6sequential_1/gru_1/gru_cell_1/ReadVariableOp_3:value:0<sequential_1/gru_1/gru_cell_1/strided_slice_3/stack:output:0>sequential_1/gru_1/gru_cell_1/strided_slice_3/stack_1:output:0>sequential_1/gru_1/gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*

begin_maskÂ
%sequential_1/gru_1/gru_cell_1/BiasAddBiasAdd.sequential_1/gru_1/gru_cell_1/MatMul:product:06sequential_1/gru_1/gru_cell_1/strided_slice_3:output:0*
T0*
_output_shapes
:	Ł
.sequential_1/gru_1/gru_cell_1/ReadVariableOp_4ReadVariableOp7sequential_1_gru_1_gru_cell_1_readvariableop_3_resource*
_output_shapes	
:*
dtype0~
3sequential_1/gru_1/gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
5sequential_1/gru_1/gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5sequential_1/gru_1/gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ć
-sequential_1/gru_1/gru_cell_1/strided_slice_4StridedSlice6sequential_1/gru_1/gru_cell_1/ReadVariableOp_4:value:0<sequential_1/gru_1/gru_cell_1/strided_slice_4/stack:output:0>sequential_1/gru_1/gru_cell_1/strided_slice_4/stack_1:output:0>sequential_1/gru_1/gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:Ć
'sequential_1/gru_1/gru_cell_1/BiasAdd_1BiasAdd0sequential_1/gru_1/gru_cell_1/MatMul_1:product:06sequential_1/gru_1/gru_cell_1/strided_slice_4:output:0*
T0*
_output_shapes
:	Ł
.sequential_1/gru_1/gru_cell_1/ReadVariableOp_5ReadVariableOp7sequential_1_gru_1_gru_cell_1_readvariableop_3_resource*
_output_shapes	
:*
dtype0~
3sequential_1/gru_1/gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:
5sequential_1/gru_1/gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
5sequential_1/gru_1/gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
-sequential_1/gru_1/gru_cell_1/strided_slice_5StridedSlice6sequential_1/gru_1/gru_cell_1/ReadVariableOp_5:value:0<sequential_1/gru_1/gru_cell_1/strided_slice_5/stack:output:0>sequential_1/gru_1/gru_cell_1/strided_slice_5/stack_1:output:0>sequential_1/gru_1/gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*
end_maskĆ
'sequential_1/gru_1/gru_cell_1/BiasAdd_2BiasAdd0sequential_1/gru_1/gru_cell_1/MatMul_2:product:06sequential_1/gru_1/gru_cell_1/strided_slice_5:output:0*
T0*
_output_shapes
:	¨
.sequential_1/gru_1/gru_cell_1/ReadVariableOp_6ReadVariableOp7sequential_1_gru_1_gru_cell_1_readvariableop_6_resource* 
_output_shapes
:
*
dtype0
3sequential_1/gru_1/gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        
5sequential_1/gru_1/gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
5sequential_1/gru_1/gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
-sequential_1/gru_1/gru_cell_1/strided_slice_6StridedSlice6sequential_1/gru_1/gru_cell_1/ReadVariableOp_6:value:0<sequential_1/gru_1/gru_cell_1/strided_slice_6/stack:output:0>sequential_1/gru_1/gru_cell_1/strided_slice_6/stack_1:output:0>sequential_1/gru_1/gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskľ
5sequential_1/gru_1/gru_cell_1/MatMul_3/ReadVariableOpReadVariableOp>sequential_1_gru_1_gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0Ń
&sequential_1/gru_1/gru_cell_1/MatMul_3MatMul=sequential_1/gru_1/gru_cell_1/MatMul_3/ReadVariableOp:value:06sequential_1/gru_1/gru_cell_1/strided_slice_6:output:0*
T0*
_output_shapes
:	¨
.sequential_1/gru_1/gru_cell_1/ReadVariableOp_7ReadVariableOp7sequential_1_gru_1_gru_cell_1_readvariableop_6_resource* 
_output_shapes
:
*
dtype0
3sequential_1/gru_1/gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       
5sequential_1/gru_1/gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
5sequential_1/gru_1/gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
-sequential_1/gru_1/gru_cell_1/strided_slice_7StridedSlice6sequential_1/gru_1/gru_cell_1/ReadVariableOp_7:value:0<sequential_1/gru_1/gru_cell_1/strided_slice_7/stack:output:0>sequential_1/gru_1/gru_cell_1/strided_slice_7/stack_1:output:0>sequential_1/gru_1/gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskľ
5sequential_1/gru_1/gru_cell_1/MatMul_4/ReadVariableOpReadVariableOp>sequential_1_gru_1_gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0Ń
&sequential_1/gru_1/gru_cell_1/MatMul_4MatMul=sequential_1/gru_1/gru_cell_1/MatMul_4/ReadVariableOp:value:06sequential_1/gru_1/gru_cell_1/strided_slice_7:output:0*
T0*
_output_shapes
:	ś
!sequential_1/gru_1/gru_cell_1/addAddV2.sequential_1/gru_1/gru_cell_1/BiasAdd:output:00sequential_1/gru_1/gru_cell_1/MatMul_3:product:0*
T0*
_output_shapes
:	
%sequential_1/gru_1/gru_cell_1/SigmoidSigmoid%sequential_1/gru_1/gru_cell_1/add:z:0*
T0*
_output_shapes
:	ş
#sequential_1/gru_1/gru_cell_1/add_1AddV20sequential_1/gru_1/gru_cell_1/BiasAdd_1:output:00sequential_1/gru_1/gru_cell_1/MatMul_4:product:0*
T0*
_output_shapes
:	
'sequential_1/gru_1/gru_cell_1/Sigmoid_1Sigmoid'sequential_1/gru_1/gru_cell_1/add_1:z:0*
T0*
_output_shapes
:	°
0sequential_1/gru_1/gru_cell_1/mul/ReadVariableOpReadVariableOp>sequential_1_gru_1_gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0š
!sequential_1/gru_1/gru_cell_1/mulMul+sequential_1/gru_1/gru_cell_1/Sigmoid_1:y:08sequential_1/gru_1/gru_cell_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:	¨
.sequential_1/gru_1/gru_cell_1/ReadVariableOp_8ReadVariableOp7sequential_1_gru_1_gru_cell_1_readvariableop_6_resource* 
_output_shapes
:
*
dtype0
3sequential_1/gru_1/gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       
5sequential_1/gru_1/gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
5sequential_1/gru_1/gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
-sequential_1/gru_1/gru_cell_1/strided_slice_8StridedSlice6sequential_1/gru_1/gru_cell_1/ReadVariableOp_8:value:0<sequential_1/gru_1/gru_cell_1/strided_slice_8/stack:output:0>sequential_1/gru_1/gru_cell_1/strided_slice_8/stack_1:output:0>sequential_1/gru_1/gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskš
&sequential_1/gru_1/gru_cell_1/MatMul_5MatMul%sequential_1/gru_1/gru_cell_1/mul:z:06sequential_1/gru_1/gru_cell_1/strided_slice_8:output:0*
T0*
_output_shapes
:	ş
#sequential_1/gru_1/gru_cell_1/add_2AddV20sequential_1/gru_1/gru_cell_1/BiasAdd_2:output:00sequential_1/gru_1/gru_cell_1/MatMul_5:product:0*
T0*
_output_shapes
:	}
"sequential_1/gru_1/gru_cell_1/TanhTanh'sequential_1/gru_1/gru_cell_1/add_2:z:0*
T0*
_output_shapes
:	˛
2sequential_1/gru_1/gru_cell_1/mul_1/ReadVariableOpReadVariableOp>sequential_1_gru_1_gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0ť
#sequential_1/gru_1/gru_cell_1/mul_1Mul)sequential_1/gru_1/gru_cell_1/Sigmoid:y:0:sequential_1/gru_1/gru_cell_1/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	h
#sequential_1/gru_1/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ť
!sequential_1/gru_1/gru_cell_1/subSub,sequential_1/gru_1/gru_cell_1/sub/x:output:0)sequential_1/gru_1/gru_cell_1/Sigmoid:y:0*
T0*
_output_shapes
:	Ł
#sequential_1/gru_1/gru_cell_1/mul_2Mul%sequential_1/gru_1/gru_cell_1/sub:z:0&sequential_1/gru_1/gru_cell_1/Tanh:y:0*
T0*
_output_shapes
:	¨
#sequential_1/gru_1/gru_cell_1/add_3AddV2'sequential_1/gru_1/gru_cell_1/mul_1:z:0'sequential_1/gru_1/gru_cell_1/mul_2:z:0*
T0*
_output_shapes
:	
0sequential_1/gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ď
"sequential_1/gru_1/TensorArrayV2_1TensorListReserve9sequential_1/gru_1/TensorArrayV2_1/element_shape:output:0)sequential_1/gru_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇY
sequential_1/gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : Ą
!sequential_1/gru_1/ReadVariableOpReadVariableOp>sequential_1_gru_1_gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0v
+sequential_1/gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙g
%sequential_1/gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
sequential_1/gru_1/whileWhile.sequential_1/gru_1/while/loop_counter:output:04sequential_1/gru_1/while/maximum_iterations:output:0 sequential_1/gru_1/time:output:0+sequential_1/gru_1/TensorArrayV2_1:handle:0)sequential_1/gru_1/ReadVariableOp:value:0)sequential_1/gru_1/strided_slice:output:0Jsequential_1/gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:05sequential_1_gru_1_gru_cell_1_readvariableop_resource7sequential_1_gru_1_gru_cell_1_readvariableop_3_resource7sequential_1_gru_1_gru_cell_1_readvariableop_6_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *0
body(R&
$sequential_1_gru_1_while_body_289180*0
cond(R&
$sequential_1_gru_1_while_cond_289179*0
output_shapes
: : : : :	: : : : : *
parallel_iterations 
Csequential_1/gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ü
5sequential_1/gru_1/TensorArrayV2Stack/TensorListStackTensorListStack!sequential_1/gru_1/while:output:3Lsequential_1/gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0{
(sequential_1/gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙t
*sequential_1/gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: t
*sequential_1/gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ţ
"sequential_1/gru_1/strided_slice_2StridedSlice>sequential_1/gru_1/TensorArrayV2Stack/TensorListStack:tensor:01sequential_1/gru_1/strided_slice_2/stack:output:03sequential_1/gru_1/strided_slice_2/stack_1:output:03sequential_1/gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_maskx
#sequential_1/gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Đ
sequential_1/gru_1/transpose_1	Transpose>sequential_1/gru_1/TensorArrayV2Stack/TensorListStack:tensor:0,sequential_1/gru_1/transpose_1/perm:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙n
sequential_1/gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    š
#sequential_1/gru_1/AssignVariableOpAssignVariableOp>sequential_1_gru_1_gru_cell_1_matmul_3_readvariableop_resource!sequential_1/gru_1/while:output:4"^sequential_1/gru_1/ReadVariableOp6^sequential_1/gru_1/gru_cell_1/MatMul_3/ReadVariableOp6^sequential_1/gru_1/gru_cell_1/MatMul_4/ReadVariableOp1^sequential_1/gru_1/gru_cell_1/mul/ReadVariableOp3^sequential_1/gru_1/gru_cell_1/mul_1/ReadVariableOp*
_output_shapes
 *
dtype0Ľ
-sequential_1/dense_1/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_1_tensordot_readvariableop_resource*
_output_shapes
:	V*
dtype0m
#sequential_1/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
#sequential_1/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       v
$sequential_1/dense_1/Tensordot/ShapeShape"sequential_1/gru_1/transpose_1:y:0*
T0*
_output_shapes
:n
,sequential_1/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
'sequential_1/dense_1/Tensordot/GatherV2GatherV2-sequential_1/dense_1/Tensordot/Shape:output:0,sequential_1/dense_1/Tensordot/free:output:05sequential_1/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
.sequential_1/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
)sequential_1/dense_1/Tensordot/GatherV2_1GatherV2-sequential_1/dense_1/Tensordot/Shape:output:0,sequential_1/dense_1/Tensordot/axes:output:07sequential_1/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
$sequential_1/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ­
#sequential_1/dense_1/Tensordot/ProdProd0sequential_1/dense_1/Tensordot/GatherV2:output:0-sequential_1/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential_1/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ł
%sequential_1/dense_1/Tensordot/Prod_1Prod2sequential_1/dense_1/Tensordot/GatherV2_1:output:0/sequential_1/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential_1/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : đ
%sequential_1/dense_1/Tensordot/concatConcatV2,sequential_1/dense_1/Tensordot/free:output:0,sequential_1/dense_1/Tensordot/axes:output:03sequential_1/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:¸
$sequential_1/dense_1/Tensordot/stackPack,sequential_1/dense_1/Tensordot/Prod:output:0.sequential_1/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ŕ
(sequential_1/dense_1/Tensordot/transpose	Transpose"sequential_1/gru_1/transpose_1:y:0.sequential_1/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙É
&sequential_1/dense_1/Tensordot/ReshapeReshape,sequential_1/dense_1/Tensordot/transpose:y:0-sequential_1/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙É
%sequential_1/dense_1/Tensordot/MatMulMatMul/sequential_1/dense_1/Tensordot/Reshape:output:05sequential_1/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Vp
&sequential_1/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Vn
,sequential_1/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ű
'sequential_1/dense_1/Tensordot/concat_1ConcatV20sequential_1/dense_1/Tensordot/GatherV2:output:0/sequential_1/dense_1/Tensordot/Const_2:output:05sequential_1/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Â
sequential_1/dense_1/TensordotReshape/sequential_1/dense_1/Tensordot/MatMul:product:00sequential_1/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙V
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:V*
dtype0ť
sequential_1/dense_1/BiasAddBiasAdd'sequential_1/dense_1/Tensordot:output:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙Vx
IdentityIdentity%sequential_1/dense_1/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙VÄ
NoOpNoOp,^sequential_1/dense_1/BiasAdd/ReadVariableOp.^sequential_1/dense_1/Tensordot/ReadVariableOp*^sequential_1/embedding_1/embedding_lookup$^sequential_1/gru_1/AssignVariableOp"^sequential_1/gru_1/ReadVariableOp6^sequential_1/gru_1/gru_cell_1/MatMul_3/ReadVariableOp6^sequential_1/gru_1/gru_cell_1/MatMul_4/ReadVariableOp-^sequential_1/gru_1/gru_cell_1/ReadVariableOp/^sequential_1/gru_1/gru_cell_1/ReadVariableOp_1/^sequential_1/gru_1/gru_cell_1/ReadVariableOp_2/^sequential_1/gru_1/gru_cell_1/ReadVariableOp_3/^sequential_1/gru_1/gru_cell_1/ReadVariableOp_4/^sequential_1/gru_1/gru_cell_1/ReadVariableOp_5/^sequential_1/gru_1/gru_cell_1/ReadVariableOp_6/^sequential_1/gru_1/gru_cell_1/ReadVariableOp_7/^sequential_1/gru_1/gru_cell_1/ReadVariableOp_81^sequential_1/gru_1/gru_cell_1/mul/ReadVariableOp3^sequential_1/gru_1/gru_cell_1/mul_1/ReadVariableOp^sequential_1/gru_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:˙˙˙˙˙˙˙˙˙: : : : : : : 2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2^
-sequential_1/dense_1/Tensordot/ReadVariableOp-sequential_1/dense_1/Tensordot/ReadVariableOp2V
)sequential_1/embedding_1/embedding_lookup)sequential_1/embedding_1/embedding_lookup2J
#sequential_1/gru_1/AssignVariableOp#sequential_1/gru_1/AssignVariableOp2F
!sequential_1/gru_1/ReadVariableOp!sequential_1/gru_1/ReadVariableOp2n
5sequential_1/gru_1/gru_cell_1/MatMul_3/ReadVariableOp5sequential_1/gru_1/gru_cell_1/MatMul_3/ReadVariableOp2n
5sequential_1/gru_1/gru_cell_1/MatMul_4/ReadVariableOp5sequential_1/gru_1/gru_cell_1/MatMul_4/ReadVariableOp2\
,sequential_1/gru_1/gru_cell_1/ReadVariableOp,sequential_1/gru_1/gru_cell_1/ReadVariableOp2`
.sequential_1/gru_1/gru_cell_1/ReadVariableOp_1.sequential_1/gru_1/gru_cell_1/ReadVariableOp_12`
.sequential_1/gru_1/gru_cell_1/ReadVariableOp_2.sequential_1/gru_1/gru_cell_1/ReadVariableOp_22`
.sequential_1/gru_1/gru_cell_1/ReadVariableOp_3.sequential_1/gru_1/gru_cell_1/ReadVariableOp_32`
.sequential_1/gru_1/gru_cell_1/ReadVariableOp_4.sequential_1/gru_1/gru_cell_1/ReadVariableOp_42`
.sequential_1/gru_1/gru_cell_1/ReadVariableOp_5.sequential_1/gru_1/gru_cell_1/ReadVariableOp_52`
.sequential_1/gru_1/gru_cell_1/ReadVariableOp_6.sequential_1/gru_1/gru_cell_1/ReadVariableOp_62`
.sequential_1/gru_1/gru_cell_1/ReadVariableOp_7.sequential_1/gru_1/gru_cell_1/ReadVariableOp_72`
.sequential_1/gru_1/gru_cell_1/ReadVariableOp_8.sequential_1/gru_1/gru_cell_1/ReadVariableOp_82d
0sequential_1/gru_1/gru_cell_1/mul/ReadVariableOp0sequential_1/gru_1/gru_cell_1/mul/ReadVariableOp2h
2sequential_1/gru_1/gru_cell_1/mul_1/ReadVariableOp2sequential_1/gru_1/gru_cell_1/mul_1/ReadVariableOp24
sequential_1/gru_1/whilesequential_1/gru_1/while:Z V
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameembedding_1_input
Ö

(__inference_dense_1_layer_call_fn_292146

inputs
unknown:	V
	unknown_0:V
identity˘StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙V*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_290203s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙V`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ĺ
š
H__inference_sequential_1_layer_call_and_return_conditional_losses_290535

inputs%
embedding_1_290517:	V 
gru_1_290520:

gru_1_290522:	 
gru_1_290524:

gru_1_290526:	!
dense_1_290529:	V
dense_1_290531:V
identity˘dense_1/StatefulPartitionedCall˘#embedding_1/StatefulPartitionedCall˘gru_1/StatefulPartitionedCallî
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_1_290517*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_embedding_1_layer_call_and_return_conditional_losses_289938ą
gru_1/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0gru_1_290520gru_1_290522gru_1_290524gru_1_290526*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_290474
dense_1/StatefulPartitionedCallStatefulPartitionedCall&gru_1/StatefulPartitionedCall:output:0dense_1_290529dense_1_290531*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙V*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_290203{
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙VŽ
NoOpNoOp ^dense_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:˙˙˙˙˙˙˙˙˙: : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
č

gru_1_while_cond_291007(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2(
$gru_1_while_less_gru_1_strided_slice@
<gru_1_while_gru_1_while_cond_291007___redundant_placeholder0@
<gru_1_while_gru_1_while_cond_291007___redundant_placeholder1@
<gru_1_while_gru_1_while_cond_291007___redundant_placeholder2@
<gru_1_while_gru_1_while_cond_291007___redundant_placeholder3
gru_1_while_identity
x
gru_1/while/LessLessgru_1_while_placeholder$gru_1_while_less_gru_1_strided_slice*
T0*
_output_shapes
: W
gru_1/while/IdentityIdentitygru_1/while/Less:z:0*
T0
*
_output_shapes
: "5
gru_1_while_identitygru_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :	: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
:
Ő	
ş
-__inference_sequential_1_layer_call_fn_290227
embedding_1_input
unknown:	V
	unknown_0:

	unknown_1:	
	unknown_2:

	unknown_3:	
	unknown_4:	V
	unknown_5:V
identity˘StatefulPartitionedCallŻ
StatefulPartitionedCallStatefulPartitionedCallembedding_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙V*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_290210s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙V`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:˙˙˙˙˙˙˙˙˙: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameembedding_1_input
ś
Ő
&__inference_gru_1_layer_call_fn_291223
inputs_0
unknown:	
	unknown_0:

	unknown_1:	
	unknown_2:

identity˘StatefulPartitionedCallů
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_289910t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/0
Ó	
Ř
+__inference_gru_cell_1_layer_call_fn_292234

inputs
states_0
unknown:

	unknown_0:	
	unknown_1:

identity

identity_1˘StatefulPartitionedCallű
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:	:	*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_289799g
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
:	i

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
:	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:	:	: : : 22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0
ŠH
°
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_289544

inputs

states+
readvariableop_resource:
(
readvariableop_3_resource:	-
readvariableop_6_resource:

identity

identity_1˘ReadVariableOp˘ReadVariableOp_1˘ReadVariableOp_2˘ReadVariableOp_3˘ReadVariableOp_4˘ReadVariableOp_5˘ReadVariableOp_6˘ReadVariableOp_7˘ReadVariableOp_8h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      í
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskZ
MatMulMatMulinputsstrided_slice:output:0*
T0*
_output_shapes
:	j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask^
MatMul_1MatMulinputsstrided_slice_1:output:0*
T0*
_output_shapes
:	j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask^
MatMul_2MatMulinputsstrided_slice_2:output:0*
T0*
_output_shapes
:	g
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes	
:*
dtype0_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: b
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*

begin_maskh
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*
_output_shapes
:	g
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes	
:*
dtype0`
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Đ
strided_slice_4StridedSliceReadVariableOp_4:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:l
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*
_output_shapes
:	g
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes	
:*
dtype0`
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ŕ
strided_slice_5StridedSliceReadVariableOp_5:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*
end_maskl
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*
_output_shapes
:	l
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource* 
_output_shapes
:
*
dtype0f
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_6StridedSliceReadVariableOp_6:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask^
MatMul_3MatMulstatesstrided_slice_6:output:0*
T0*
_output_shapes
:	l
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource* 
_output_shapes
:
*
dtype0f
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_7StridedSliceReadVariableOp_7:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask^
MatMul_4MatMulstatesstrided_slice_7:output:0*
T0*
_output_shapes
:	\
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*
_output_shapes
:	E
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:	`
add_1AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*
_output_shapes
:	I
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:	K
mulMulSigmoid_1:y:0states*
T0*
_output_shapes
:	l
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource* 
_output_shapes
:
*
dtype0f
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_8StridedSliceReadVariableOp_8:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask_
MatMul_5MatMulmul:z:0strided_slice_8:output:0*
T0*
_output_shapes
:	`
add_2AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*
_output_shapes
:	A
TanhTanh	add_2:z:0*
T0*
_output_shapes
:	K
mul_1MulSigmoid:y:0states*
T0*
_output_shapes
:	J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Q
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:	I
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:	N
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:	P
IdentityIdentity	add_3:z:0^NoOp*
T0*
_output_shapes
:	R

Identity_1Identity	add_3:z:0^NoOp*
T0*
_output_shapes
:	ď
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:	:	: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_8:G C

_output_shapes
:	
 
_user_specified_nameinputs:GC

_output_shapes
:	
 
_user_specified_namestates
Ď

A__inference_gru_1_layer_call_and_return_conditional_losses_291471
inputs_06
"gru_cell_1_readvariableop_resource:
3
$gru_cell_1_readvariableop_3_resource:	8
$gru_cell_1_readvariableop_6_resource:
>
+gru_cell_1_matmul_3_readvariableop_resource:	
identity˘AssignVariableOp˘ReadVariableOp˘"gru_cell_1/MatMul_3/ReadVariableOp˘"gru_cell_1/MatMul_4/ReadVariableOp˘gru_cell_1/ReadVariableOp˘gru_cell_1/ReadVariableOp_1˘gru_cell_1/ReadVariableOp_2˘gru_cell_1/ReadVariableOp_3˘gru_cell_1/ReadVariableOp_4˘gru_cell_1/ReadVariableOp_5˘gru_cell_1/ReadVariableOp_6˘gru_cell_1/ReadVariableOp_7˘gru_cell_1/ReadVariableOp_8˘gru_cell_1/mul/ReadVariableOp˘gru_cell_1/mul_1/ReadVariableOp˘whilec
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          p
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙˛
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:á
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_mask~
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype0o
gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        q
 gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       q
 gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¤
gru_cell_1/strided_sliceStridedSlice!gru_cell_1/ReadVariableOp:value:0'gru_cell_1/strided_slice/stack:output:0)gru_cell_1/strided_slice/stack_1:output:0)gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_cell_1/MatMulMatMulstrided_slice_1:output:0!gru_cell_1/strided_slice:output:0*
T0*
_output_shapes
:	
gru_cell_1/ReadVariableOp_1ReadVariableOp"gru_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype0q
 gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
gru_cell_1/strided_slice_1StridedSlice#gru_cell_1/ReadVariableOp_1:value:0)gru_cell_1/strided_slice_1/stack:output:0+gru_cell_1/strided_slice_1/stack_1:output:0+gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_cell_1/MatMul_1MatMulstrided_slice_1:output:0#gru_cell_1/strided_slice_1:output:0*
T0*
_output_shapes
:	
gru_cell_1/ReadVariableOp_2ReadVariableOp"gru_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype0q
 gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        s
"gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
gru_cell_1/strided_slice_2StridedSlice#gru_cell_1/ReadVariableOp_2:value:0)gru_cell_1/strided_slice_2/stack:output:0+gru_cell_1/strided_slice_2/stack_1:output:0+gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_cell_1/MatMul_2MatMulstrided_slice_1:output:0#gru_cell_1/strided_slice_2:output:0*
T0*
_output_shapes
:	}
gru_cell_1/ReadVariableOp_3ReadVariableOp$gru_cell_1_readvariableop_3_resource*
_output_shapes	
:*
dtype0j
 gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: m
"gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_cell_1/strided_slice_3StridedSlice#gru_cell_1/ReadVariableOp_3:value:0)gru_cell_1/strided_slice_3/stack:output:0+gru_cell_1/strided_slice_3/stack_1:output:0+gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*

begin_mask
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0#gru_cell_1/strided_slice_3:output:0*
T0*
_output_shapes
:	}
gru_cell_1/ReadVariableOp_4ReadVariableOp$gru_cell_1_readvariableop_3_resource*
_output_shapes	
:*
dtype0k
 gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:m
"gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_cell_1/strided_slice_4StridedSlice#gru_cell_1/ReadVariableOp_4:value:0)gru_cell_1/strided_slice_4/stack:output:0+gru_cell_1/strided_slice_4/stack_1:output:0+gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0#gru_cell_1/strided_slice_4:output:0*
T0*
_output_shapes
:	}
gru_cell_1/ReadVariableOp_5ReadVariableOp$gru_cell_1_readvariableop_3_resource*
_output_shapes	
:*
dtype0k
 gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:l
"gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: l
"gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_cell_1/strided_slice_5StridedSlice#gru_cell_1/ReadVariableOp_5:value:0)gru_cell_1/strided_slice_5/stack:output:0+gru_cell_1/strided_slice_5/stack_1:output:0+gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*
end_mask
gru_cell_1/BiasAdd_2BiasAddgru_cell_1/MatMul_2:product:0#gru_cell_1/strided_slice_5:output:0*
T0*
_output_shapes
:	
gru_cell_1/ReadVariableOp_6ReadVariableOp$gru_cell_1_readvariableop_6_resource* 
_output_shapes
:
*
dtype0q
 gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
gru_cell_1/strided_slice_6StridedSlice#gru_cell_1/ReadVariableOp_6:value:0)gru_cell_1/strided_slice_6/stack:output:0+gru_cell_1/strided_slice_6/stack_1:output:0+gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
"gru_cell_1/MatMul_3/ReadVariableOpReadVariableOp+gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_1/MatMul_3MatMul*gru_cell_1/MatMul_3/ReadVariableOp:value:0#gru_cell_1/strided_slice_6:output:0*
T0*
_output_shapes
:	
gru_cell_1/ReadVariableOp_7ReadVariableOp$gru_cell_1_readvariableop_6_resource* 
_output_shapes
:
*
dtype0q
 gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
gru_cell_1/strided_slice_7StridedSlice#gru_cell_1/ReadVariableOp_7:value:0)gru_cell_1/strided_slice_7/stack:output:0+gru_cell_1/strided_slice_7/stack_1:output:0+gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
"gru_cell_1/MatMul_4/ReadVariableOpReadVariableOp+gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_1/MatMul_4MatMul*gru_cell_1/MatMul_4/ReadVariableOp:value:0#gru_cell_1/strided_slice_7:output:0*
T0*
_output_shapes
:	}
gru_cell_1/addAddV2gru_cell_1/BiasAdd:output:0gru_cell_1/MatMul_3:product:0*
T0*
_output_shapes
:	[
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*
_output_shapes
:	
gru_cell_1/add_1AddV2gru_cell_1/BiasAdd_1:output:0gru_cell_1/MatMul_4:product:0*
T0*
_output_shapes
:	_
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*
_output_shapes
:	
gru_cell_1/mul/ReadVariableOpReadVariableOp+gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0%gru_cell_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:	
gru_cell_1/ReadVariableOp_8ReadVariableOp$gru_cell_1_readvariableop_6_resource* 
_output_shapes
:
*
dtype0q
 gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       s
"gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        s
"gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
gru_cell_1/strided_slice_8StridedSlice#gru_cell_1/ReadVariableOp_8:value:0)gru_cell_1/strided_slice_8/stack:output:0+gru_cell_1/strided_slice_8/stack_1:output:0+gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
gru_cell_1/MatMul_5MatMulgru_cell_1/mul:z:0#gru_cell_1/strided_slice_8:output:0*
T0*
_output_shapes
:	
gru_cell_1/add_2AddV2gru_cell_1/BiasAdd_2:output:0gru_cell_1/MatMul_5:product:0*
T0*
_output_shapes
:	W
gru_cell_1/TanhTanhgru_cell_1/add_2:z:0*
T0*
_output_shapes
:	
gru_cell_1/mul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0'gru_cell_1/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	U
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?r
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*
_output_shapes
:	j
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Tanh:y:0*
T0*
_output_shapes
:	o
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*
_output_shapes
:	n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ś
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : {
ReadVariableOpReadVariableOp+gru_cell_1_matmul_3_readvariableop_resource*
_output_shapes
:	*
dtype0c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ľ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0ReadVariableOp:value:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_1_readvariableop_resource$gru_cell_1_readvariableop_3_resource$gru_cell_1_readvariableop_6_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_291346*
condR
while_cond_291345*0
output_shapes
: : : : :	: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ă
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:˙
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    Ą
AssignVariableOpAssignVariableOp+gru_cell_1_matmul_3_readvariableop_resourcewhile:output:4^ReadVariableOp#^gru_cell_1/MatMul_3/ReadVariableOp#^gru_cell_1/MatMul_4/ReadVariableOp^gru_cell_1/mul/ReadVariableOp ^gru_cell_1/mul_1/ReadVariableOp*
_output_shapes
 *
dtype0c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp^AssignVariableOp^ReadVariableOp#^gru_cell_1/MatMul_3/ReadVariableOp#^gru_cell_1/MatMul_4/ReadVariableOp^gru_cell_1/ReadVariableOp^gru_cell_1/ReadVariableOp_1^gru_cell_1/ReadVariableOp_2^gru_cell_1/ReadVariableOp_3^gru_cell_1/ReadVariableOp_4^gru_cell_1/ReadVariableOp_5^gru_cell_1/ReadVariableOp_6^gru_cell_1/ReadVariableOp_7^gru_cell_1/ReadVariableOp_8^gru_cell_1/mul/ReadVariableOp ^gru_cell_1/mul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙: : : : 2$
AssignVariableOpAssignVariableOp2 
ReadVariableOpReadVariableOp2H
"gru_cell_1/MatMul_3/ReadVariableOp"gru_cell_1/MatMul_3/ReadVariableOp2H
"gru_cell_1/MatMul_4/ReadVariableOp"gru_cell_1/MatMul_4/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2:
gru_cell_1/ReadVariableOp_1gru_cell_1/ReadVariableOp_12:
gru_cell_1/ReadVariableOp_2gru_cell_1/ReadVariableOp_22:
gru_cell_1/ReadVariableOp_3gru_cell_1/ReadVariableOp_32:
gru_cell_1/ReadVariableOp_4gru_cell_1/ReadVariableOp_42:
gru_cell_1/ReadVariableOp_5gru_cell_1/ReadVariableOp_52:
gru_cell_1/ReadVariableOp_6gru_cell_1/ReadVariableOp_62:
gru_cell_1/ReadVariableOp_7gru_cell_1/ReadVariableOp_72:
gru_cell_1/ReadVariableOp_8gru_cell_1/ReadVariableOp_82>
gru_cell_1/mul/ReadVariableOpgru_cell_1/mul/ReadVariableOp2B
gru_cell_1/mul_1/ReadVariableOpgru_cell_1/mul_1/ReadVariableOp2
whilewhile:V R
,
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/0
¨	
Ľ
G__inference_embedding_1_layer_call_and_return_conditional_losses_289938

inputs*
embedding_lookup_289932:	V
identity˘embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ź
embedding_lookupResourceGatherembedding_lookup_289932Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/289932*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0Ł
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/289932*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙x
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:˙˙˙˙˙˙˙˙˙: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ňM

F__inference_gru_cell_1_layer_call_and_return_conditional_losses_292314

inputs
states_0+
readvariableop_resource:
(
readvariableop_3_resource:	-
readvariableop_6_resource:

identity

identity_1˘MatMul_3/ReadVariableOp˘MatMul_4/ReadVariableOp˘ReadVariableOp˘ReadVariableOp_1˘ReadVariableOp_2˘ReadVariableOp_3˘ReadVariableOp_4˘ReadVariableOp_5˘ReadVariableOp_6˘ReadVariableOp_7˘ReadVariableOp_8˘mul/ReadVariableOp˘mul_1/ReadVariableOph
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      í
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskZ
MatMulMatMulinputsstrided_slice:output:0*
T0*
_output_shapes
:	j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask^
MatMul_1MatMulinputsstrided_slice_1:output:0*
T0*
_output_shapes
:	j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask^
MatMul_2MatMulinputsstrided_slice_2:output:0*
T0*
_output_shapes
:	g
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes	
:*
dtype0_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: b
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*

begin_maskh
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*
_output_shapes
:	g
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes	
:*
dtype0`
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Đ
strided_slice_4StridedSliceReadVariableOp_4:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:l
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*
_output_shapes
:	g
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes	
:*
dtype0`
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ŕ
strided_slice_5StridedSliceReadVariableOp_5:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*
end_maskl
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*
_output_shapes
:	l
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource* 
_output_shapes
:
*
dtype0f
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_6StridedSliceReadVariableOp_6:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskZ
MatMul_3/ReadVariableOpReadVariableOpstates_0*
_output_shapes
:*
dtype0w
MatMul_3BatchMatMulV2MatMul_3/ReadVariableOp:value:0strided_slice_6:output:0*
T0*
_output_shapes
:l
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource* 
_output_shapes
:
*
dtype0f
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_7StridedSliceReadVariableOp_7:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskZ
MatMul_4/ReadVariableOpReadVariableOpstates_0*
_output_shapes
:*
dtype0w
MatMul_4BatchMatMulV2MatMul_4/ReadVariableOp:value:0strided_slice_7:output:0*
T0*
_output_shapes
:T
addAddV2BiasAdd:output:0MatMul_3:output:0*
T0*
_output_shapes
:>
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:X
add_1AddV2BiasAdd_1:output:0MatMul_4:output:0*
T0*
_output_shapes
:B
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:U
mul/ReadVariableOpReadVariableOpstates_0*
_output_shapes
:*
dtype0X
mulMulSigmoid_1:y:0mul/ReadVariableOp:value:0*
T0*
_output_shapes
:l
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource* 
_output_shapes
:
*
dtype0f
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_8StridedSliceReadVariableOp_8:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask_
MatMul_5BatchMatMulV2mul:z:0strided_slice_8:output:0*
T0*
_output_shapes
:X
add_2AddV2BiasAdd_2:output:0MatMul_5:output:0*
T0*
_output_shapes
::
TanhTanh	add_2:z:0*
T0*
_output_shapes
:W
mul_1/ReadVariableOpReadVariableOpstates_0*
_output_shapes
:*
dtype0Z
mul_1MulSigmoid:y:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?J
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:B
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:G
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:I
IdentityIdentity	add_3:z:0^NoOp*
T0*
_output_shapes
:K

Identity_1Identity	add_3:z:0^NoOp*
T0*
_output_shapes
:Ď
NoOpNoOp^MatMul_3/ReadVariableOp^MatMul_4/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8^mul/ReadVariableOp^mul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:	:	: : : 22
MatMul_3/ReadVariableOpMatMul_3/ReadVariableOp22
MatMul_4/ReadVariableOpMatMul_4/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82(
mul/ReadVariableOpmul/ReadVariableOp2,
mul_1/ReadVariableOpmul_1/ReadVariableOp:G C

_output_shapes
:	
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0"ŰL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Â
serving_defaultŽ
O
embedding_1_input:
#serving_default_embedding_1_input:0˙˙˙˙˙˙˙˙˙?
dense_14
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙Vtensorflow/serving/predict:Žo
Ě
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature

signatures"
_tf_keras_sequential
ľ

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
Ú
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
ť

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_layer
J
0
$1
%2
&3
4
5"
trackable_list_wrapper
J
0
$1
%2
&3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
Ę
'non_trainable_variables

(layers
)metrics
*layer_regularization_losses
+layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
2˙
-__inference_sequential_1_layer_call_fn_290227
-__inference_sequential_1_layer_call_fn_290632
-__inference_sequential_1_layer_call_fn_290651
-__inference_sequential_1_layer_call_fn_290571Ŕ
ˇ˛ł
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
kwonlydefaultsŞ 
annotationsŞ *
 
î2ë
H__inference_sequential_1_layer_call_and_return_conditional_losses_290905
H__inference_sequential_1_layer_call_and_return_conditional_losses_291159
H__inference_sequential_1_layer_call_and_return_conditional_losses_290592
H__inference_sequential_1_layer_call_and_return_conditional_losses_290613Ŕ
ˇ˛ł
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
kwonlydefaultsŞ 
annotationsŞ *
 
ÖBÓ
!__inference__wrapped_model_289331embedding_1_input"
˛
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
,
,serving_default"
signature_map
):'	V2embedding_1/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_embedding_1_layer_call_fn_291187˘
˛
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
annotationsŞ *
 
ń2î
G__inference_embedding_1_layer_call_and_return_conditional_losses_291197˘
˛
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
annotationsŞ *
 
č

$kernel
%recurrent_kernel
&bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6_random_generator
7__call__
*8&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
$0
%1
&2"
trackable_list_wrapper
5
$0
%1
&2"
trackable_list_wrapper
 "
trackable_list_wrapper
š

9states
:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
ű2ř
&__inference_gru_1_layer_call_fn_291210
&__inference_gru_1_layer_call_fn_291223
&__inference_gru_1_layer_call_fn_291236
&__inference_gru_1_layer_call_fn_291249Ő
Ě˛Č
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ç2ä
A__inference_gru_1_layer_call_and_return_conditional_losses_291471
A__inference_gru_1_layer_call_and_return_conditional_losses_291693
A__inference_gru_1_layer_call_and_return_conditional_losses_291915
A__inference_gru_1_layer_call_and_return_conditional_losses_292137Ő
Ě˛Č
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
!:	V2dense_1/kernel
:V2dense_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
Ň2Ď
(__inference_dense_1_layer_call_fn_292146˘
˛
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
annotationsŞ *
 
í2ę
C__inference_dense_1_layer_call_and_return_conditional_losses_292176˘
˛
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
annotationsŞ *
 
+:)
2gru_1/gru_cell_1/kernel
5:3
2!gru_1/gru_cell_1/recurrent_kernel
$:"2gru_1/gru_cell_1/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ŐBŇ
$__inference_signature_wrapper_291180embedding_1_input"
˛
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
annotationsŞ *
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
5
$0
%1
&2"
trackable_list_wrapper
5
$0
%1
&2"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
2	variables
3trainable_variables
4regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
ř2ő
+__inference_gru_cell_1_layer_call_fn_292190
+__inference_gru_cell_1_layer_call_fn_292204
+__inference_gru_cell_1_layer_call_fn_292219
+__inference_gru_cell_1_layer_call_fn_292234ž
ľ˛ą
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ä2á
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_292314
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_292389
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_292464
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_292544ž
ľ˛ą
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
'
I0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
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
!:	2gru_1/VariableĄ
!__inference__wrapped_model_289331|$&%I:˘7
0˘-
+(
embedding_1_input˙˙˙˙˙˙˙˙˙
Ş "5Ş2
0
dense_1%"
dense_1˙˙˙˙˙˙˙˙˙VŹ
C__inference_dense_1_layer_call_and_return_conditional_losses_292176e4˘1
*˘'
%"
inputs˙˙˙˙˙˙˙˙˙
Ş ")˘&

0˙˙˙˙˙˙˙˙˙V
 
(__inference_dense_1_layer_call_fn_292146X4˘1
*˘'
%"
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙VŤ
G__inference_embedding_1_layer_call_and_return_conditional_losses_291197`/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙
Ş "*˘'
 
0˙˙˙˙˙˙˙˙˙
 
,__inference_embedding_1_layer_call_fn_291187S/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙Ŕ
A__inference_gru_1_layer_call_and_return_conditional_losses_291471{$&%IG˘D
=˘:
,)
'$
inputs/0˙˙˙˙˙˙˙˙˙

 
p 

 
Ş "*˘'
 
0˙˙˙˙˙˙˙˙˙
 Ŕ
A__inference_gru_1_layer_call_and_return_conditional_losses_291693{$&%IG˘D
=˘:
,)
'$
inputs/0˙˙˙˙˙˙˙˙˙

 
p

 
Ş "*˘'
 
0˙˙˙˙˙˙˙˙˙
 š
A__inference_gru_1_layer_call_and_return_conditional_losses_291915t$&%I@˘=
6˘3
%"
inputs˙˙˙˙˙˙˙˙˙

 
p 

 
Ş "*˘'
 
0˙˙˙˙˙˙˙˙˙
 š
A__inference_gru_1_layer_call_and_return_conditional_losses_292137t$&%I@˘=
6˘3
%"
inputs˙˙˙˙˙˙˙˙˙

 
p

 
Ş "*˘'
 
0˙˙˙˙˙˙˙˙˙
 
&__inference_gru_1_layer_call_fn_291210nI$&%G˘D
=˘:
,)
'$
inputs/0˙˙˙˙˙˙˙˙˙

 
p 

 
Ş "˙˙˙˙˙˙˙˙˙
&__inference_gru_1_layer_call_fn_291223nI$&%G˘D
=˘:
,)
'$
inputs/0˙˙˙˙˙˙˙˙˙

 
p

 
Ş "˙˙˙˙˙˙˙˙˙
&__inference_gru_1_layer_call_fn_291236g$&%I@˘=
6˘3
%"
inputs˙˙˙˙˙˙˙˙˙

 
p 

 
Ş "˙˙˙˙˙˙˙˙˙
&__inference_gru_1_layer_call_fn_291249g$&%I@˘=
6˘3
%"
inputs˙˙˙˙˙˙˙˙˙

 
p

 
Ş "˙˙˙˙˙˙˙˙˙ó
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_292314¨$&%k˘h
a˘^

inputs	
>˘;
96	"˘
ú	


jstates/0VariableSpec 
p 
Ş "4˘1
*˘'

0/0


0/1/0
 â
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_292389$&%L˘I
B˘?

inputs	
˘

states/0	
p 
Ş "B˘?
8˘5

0/0	


0/1/0	
 â
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_292464$&%L˘I
B˘?

inputs	
˘

states/0	
p
Ş "B˘?
8˘5

0/0	


0/1/0	
 ó
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_292544¨$&%k˘h
a˘^

inputs	
>˘;
96	"˘
ú	


jstates/0VariableSpec 
p
Ş "4˘1
*˘'

0/0


0/1/0
 š
+__inference_gru_cell_1_layer_call_fn_292190$&%L˘I
B˘?

inputs	
˘

states/0	
p 
Ş "4˘1

0	


1/0	š
+__inference_gru_cell_1_layer_call_fn_292204$&%L˘I
B˘?

inputs	
˘

states/0	
p
Ş "4˘1

0	


1/0	Ř
+__inference_gru_cell_1_layer_call_fn_292219¨$&%k˘h
a˘^

inputs	
>˘;
96	"˘
ú	


jstates/0VariableSpec 
p 
Ş "4˘1

0	


1/0	Ř
+__inference_gru_cell_1_layer_call_fn_292234¨$&%k˘h
a˘^

inputs	
>˘;
96	"˘
ú	


jstates/0VariableSpec 
p
Ş "4˘1

0	


1/0	Ä
H__inference_sequential_1_layer_call_and_return_conditional_losses_290592x$&%IB˘?
8˘5
+(
embedding_1_input˙˙˙˙˙˙˙˙˙
p 

 
Ş ")˘&

0˙˙˙˙˙˙˙˙˙V
 Ä
H__inference_sequential_1_layer_call_and_return_conditional_losses_290613x$&%IB˘?
8˘5
+(
embedding_1_input˙˙˙˙˙˙˙˙˙
p

 
Ş ")˘&

0˙˙˙˙˙˙˙˙˙V
 š
H__inference_sequential_1_layer_call_and_return_conditional_losses_290905m$&%I7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş ")˘&

0˙˙˙˙˙˙˙˙˙V
 š
H__inference_sequential_1_layer_call_and_return_conditional_losses_291159m$&%I7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş ")˘&

0˙˙˙˙˙˙˙˙˙V
 
-__inference_sequential_1_layer_call_fn_290227k$&%IB˘?
8˘5
+(
embedding_1_input˙˙˙˙˙˙˙˙˙
p 

 
Ş "˙˙˙˙˙˙˙˙˙V
-__inference_sequential_1_layer_call_fn_290571k$&%IB˘?
8˘5
+(
embedding_1_input˙˙˙˙˙˙˙˙˙
p

 
Ş "˙˙˙˙˙˙˙˙˙V
-__inference_sequential_1_layer_call_fn_290632`$&%I7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş "˙˙˙˙˙˙˙˙˙V
-__inference_sequential_1_layer_call_fn_290651`$&%I7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş "˙˙˙˙˙˙˙˙˙Vş
$__inference_signature_wrapper_291180$&%IO˘L
˘ 
EŞB
@
embedding_1_input+(
embedding_1_input˙˙˙˙˙˙˙˙˙"5Ş2
0
dense_1%"
dense_1˙˙˙˙˙˙˙˙˙V