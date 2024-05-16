%% Load mobilenetv2 initial parameters
% Load parameters for network initialization

trainingSetup = load("C:\Users\...\mobilenetv2_info.mat");
%% Import Data
% Import training and validation data

dataFolder = 'C:\Users\...\project\TrashDataset';
categories = {'BeverageCans', 'BottleCaps', 'CigaretteButts', 'PlasticBags', 'PlasticBottlesJugs'};
imds = imageDatastore(fullfile(dataFolder, categories), 'LabelSource', 'foldernames');
tbl = countEachLabel(imds);
disp (tbl)

%% Split the dataset
[trainingSet, validationSet] = splitEachLabel(imds, 0.7, 'randomized');

countEachLabel(trainingSet)
countEachLabel(validationSet)

%% Define image augmentation object 
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);

%% Resize image data
inputSize = [224 224 3];
augimdsTrain = augmentedImageDatastore(inputSize, trainingSet, 'ColorPreprocessing', 'gray2rgb', ...
    'DataAugmentation',imageAugmenter);

disp(augimdsTrain.NumObservations)

augimdsValidation = augmentedImageDatastore(inputSize, validationSet, 'ColorPreprocessing', 'gray2rgb');

disp(augimdsValidation.NumObservations)

%% Set training options
% Specify options to use when training

opts = trainingOptions("sgdm",...
    "ExecutionEnvironment","auto",...
    "InitialLearnRate",0.0001,...
    "MaxEpochs",6,...
    "MiniBatchSize",10,...
    "Shuffle","every-epoch",...
    "ValidationFrequency",3,...
    "Plots","training-progress",...
    "ValidationData",augimdsValidation);
%% Create mobilenetv2 network via layergraph
% Create the layer graph variable to contain the network layers

lgraph = layerGraph();

%% Add layer branches

tempLayers = [
    imageInputLayer([224 224 3],"Name","input_1","Normalization","zscore","Mean",trainingSetup.input_1.Mean,"StandardDeviation",trainingSetup.input_1.StandardDeviation)
    convolution2dLayer([3 3],32,"Name","Conv1","Padding","same","Stride",[2 2],"Bias",trainingSetup.Conv1.Bias,"Weights",trainingSetup.Conv1.Weights)
    batchNormalizationLayer("Name","bn_Conv1","Epsilon",0.001,"Offset",trainingSetup.bn_Conv1.Offset,"Scale",trainingSetup.bn_Conv1.Scale,"TrainedMean",trainingSetup.bn_Conv1.TrainedMean,"TrainedVariance",trainingSetup.bn_Conv1.TrainedVariance)
    clippedReluLayer(6,"Name","Conv1_relu")
    groupedConvolution2dLayer([3 3],1,32,"Name","expanded_conv_depthwise","Padding","same","Bias",trainingSetup.expanded_conv_depthwise.Bias,"Weights",trainingSetup.expanded_conv_depthwise.Weights)
    batchNormalizationLayer("Name","expanded_conv_depthwise_BN","Epsilon",0.001,"Offset",trainingSetup.expanded_conv_depthwise_BN.Offset,"Scale",trainingSetup.expanded_conv_depthwise_BN.Scale,"TrainedMean",trainingSetup.expanded_conv_depthwise_BN.TrainedMean,"TrainedVariance",trainingSetup.expanded_conv_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","expanded_conv_depthwise_relu")
    convolution2dLayer([1 1],16,"Name","expanded_conv_project","Padding","same","Bias",trainingSetup.expanded_conv_project.Bias,"Weights",trainingSetup.expanded_conv_project.Weights)
    batchNormalizationLayer("Name","expanded_conv_project_BN","Epsilon",0.001,"Offset",trainingSetup.expanded_conv_project_BN.Offset,"Scale",trainingSetup.expanded_conv_project_BN.Scale,"TrainedMean",trainingSetup.expanded_conv_project_BN.TrainedMean,"TrainedVariance",trainingSetup.expanded_conv_project_BN.TrainedVariance)
    convolution2dLayer([1 1],96,"Name","block_1_expand","Padding","same","Bias",trainingSetup.block_1_expand.Bias,"Weights",trainingSetup.block_1_expand.Weights)
    batchNormalizationLayer("Name","block_1_expand_BN","Epsilon",0.001,"Offset",trainingSetup.block_1_expand_BN.Offset,"Scale",trainingSetup.block_1_expand_BN.Scale,"TrainedMean",trainingSetup.block_1_expand_BN.TrainedMean,"TrainedVariance",trainingSetup.block_1_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_1_expand_relu")
    groupedConvolution2dLayer([3 3],1,96,"Name","block_1_depthwise","Padding","same","Stride",[2 2],"Bias",trainingSetup.block_1_depthwise.Bias,"Weights",trainingSetup.block_1_depthwise.Weights)
    batchNormalizationLayer("Name","block_1_depthwise_BN","Epsilon",0.001,"Offset",trainingSetup.block_1_depthwise_BN.Offset,"Scale",trainingSetup.block_1_depthwise_BN.Scale,"TrainedMean",trainingSetup.block_1_depthwise_BN.TrainedMean,"TrainedVariance",trainingSetup.block_1_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_1_depthwise_relu")
    convolution2dLayer([1 1],24,"Name","block_1_project","Padding","same","Bias",trainingSetup.block_1_project.Bias,"Weights",trainingSetup.block_1_project.Weights)
    batchNormalizationLayer("Name","block_1_project_BN","Epsilon",0.001,"Offset",trainingSetup.block_1_project_BN.Offset,"Scale",trainingSetup.block_1_project_BN.Scale,"TrainedMean",trainingSetup.block_1_project_BN.TrainedMean,"TrainedVariance",trainingSetup.block_1_project_BN.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],144,"Name","block_2_expand","Padding","same","Bias",trainingSetup.block_2_expand.Bias,"Weights",trainingSetup.block_2_expand.Weights)
    batchNormalizationLayer("Name","block_2_expand_BN","Epsilon",0.001,"Offset",trainingSetup.block_2_expand_BN.Offset,"Scale",trainingSetup.block_2_expand_BN.Scale,"TrainedMean",trainingSetup.block_2_expand_BN.TrainedMean,"TrainedVariance",trainingSetup.block_2_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_2_expand_relu")
    groupedConvolution2dLayer([3 3],1,144,"Name","block_2_depthwise","Padding","same","Bias",trainingSetup.block_2_depthwise.Bias,"Weights",trainingSetup.block_2_depthwise.Weights)
    batchNormalizationLayer("Name","block_2_depthwise_BN","Epsilon",0.001,"Offset",trainingSetup.block_2_depthwise_BN.Offset,"Scale",trainingSetup.block_2_depthwise_BN.Scale,"TrainedMean",trainingSetup.block_2_depthwise_BN.TrainedMean,"TrainedVariance",trainingSetup.block_2_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_2_depthwise_relu")
    convolution2dLayer([1 1],24,"Name","block_2_project","Padding","same","Bias",trainingSetup.block_2_project.Bias,"Weights",trainingSetup.block_2_project.Weights)
    batchNormalizationLayer("Name","block_2_project_BN","Epsilon",0.001,"Offset",trainingSetup.block_2_project_BN.Offset,"Scale",trainingSetup.block_2_project_BN.Scale,"TrainedMean",trainingSetup.block_2_project_BN.TrainedMean,"TrainedVariance",trainingSetup.block_2_project_BN.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block_2_add")
    convolution2dLayer([1 1],144,"Name","block_3_expand","Padding","same","Bias",trainingSetup.block_3_expand.Bias,"Weights",trainingSetup.block_3_expand.Weights)
    batchNormalizationLayer("Name","block_3_expand_BN","Epsilon",0.001,"Offset",trainingSetup.block_3_expand_BN.Offset,"Scale",trainingSetup.block_3_expand_BN.Scale,"TrainedMean",trainingSetup.block_3_expand_BN.TrainedMean,"TrainedVariance",trainingSetup.block_3_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_3_expand_relu")
    groupedConvolution2dLayer([3 3],1,144,"Name","block_3_depthwise","Padding","same","Stride",[2 2],"Bias",trainingSetup.block_3_depthwise.Bias,"Weights",trainingSetup.block_3_depthwise.Weights)
    batchNormalizationLayer("Name","block_3_depthwise_BN","Epsilon",0.001,"Offset",trainingSetup.block_3_depthwise_BN.Offset,"Scale",trainingSetup.block_3_depthwise_BN.Scale,"TrainedMean",trainingSetup.block_3_depthwise_BN.TrainedMean,"TrainedVariance",trainingSetup.block_3_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_3_depthwise_relu")
    convolution2dLayer([1 1],32,"Name","block_3_project","Padding","same","Bias",trainingSetup.block_3_project.Bias,"Weights",trainingSetup.block_3_project.Weights)
    batchNormalizationLayer("Name","block_3_project_BN","Epsilon",0.001,"Offset",trainingSetup.block_3_project_BN.Offset,"Scale",trainingSetup.block_3_project_BN.Scale,"TrainedMean",trainingSetup.block_3_project_BN.TrainedMean,"TrainedVariance",trainingSetup.block_3_project_BN.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","block_4_expand","Padding","same","Bias",trainingSetup.block_4_expand.Bias,"Weights",trainingSetup.block_4_expand.Weights)
    batchNormalizationLayer("Name","block_4_expand_BN","Epsilon",0.001,"Offset",trainingSetup.block_4_expand_BN.Offset,"Scale",trainingSetup.block_4_expand_BN.Scale,"TrainedMean",trainingSetup.block_4_expand_BN.TrainedMean,"TrainedVariance",trainingSetup.block_4_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_4_expand_relu")
    groupedConvolution2dLayer([3 3],1,192,"Name","block_4_depthwise","Padding","same","Bias",trainingSetup.block_4_depthwise.Bias,"Weights",trainingSetup.block_4_depthwise.Weights)
    batchNormalizationLayer("Name","block_4_depthwise_BN","Epsilon",0.001,"Offset",trainingSetup.block_4_depthwise_BN.Offset,"Scale",trainingSetup.block_4_depthwise_BN.Scale,"TrainedMean",trainingSetup.block_4_depthwise_BN.TrainedMean,"TrainedVariance",trainingSetup.block_4_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_4_depthwise_relu")
    convolution2dLayer([1 1],32,"Name","block_4_project","Padding","same","Bias",trainingSetup.block_4_project.Bias,"Weights",trainingSetup.block_4_project.Weights)
    batchNormalizationLayer("Name","block_4_project_BN","Epsilon",0.001,"Offset",trainingSetup.block_4_project_BN.Offset,"Scale",trainingSetup.block_4_project_BN.Scale,"TrainedMean",trainingSetup.block_4_project_BN.TrainedMean,"TrainedVariance",trainingSetup.block_4_project_BN.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","block_4_add");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","block_5_expand","Padding","same","Bias",trainingSetup.block_5_expand.Bias,"Weights",trainingSetup.block_5_expand.Weights)
    batchNormalizationLayer("Name","block_5_expand_BN","Epsilon",0.001,"Offset",trainingSetup.block_5_expand_BN.Offset,"Scale",trainingSetup.block_5_expand_BN.Scale,"TrainedMean",trainingSetup.block_5_expand_BN.TrainedMean,"TrainedVariance",trainingSetup.block_5_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_5_expand_relu")
    groupedConvolution2dLayer([3 3],1,192,"Name","block_5_depthwise","Padding","same","Bias",trainingSetup.block_5_depthwise.Bias,"Weights",trainingSetup.block_5_depthwise.Weights)
    batchNormalizationLayer("Name","block_5_depthwise_BN","Epsilon",0.001,"Offset",trainingSetup.block_5_depthwise_BN.Offset,"Scale",trainingSetup.block_5_depthwise_BN.Scale,"TrainedMean",trainingSetup.block_5_depthwise_BN.TrainedMean,"TrainedVariance",trainingSetup.block_5_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_5_depthwise_relu")
    convolution2dLayer([1 1],32,"Name","block_5_project","Padding","same","Bias",trainingSetup.block_5_project.Bias,"Weights",trainingSetup.block_5_project.Weights)
    batchNormalizationLayer("Name","block_5_project_BN","Epsilon",0.001,"Offset",trainingSetup.block_5_project_BN.Offset,"Scale",trainingSetup.block_5_project_BN.Scale,"TrainedMean",trainingSetup.block_5_project_BN.TrainedMean,"TrainedVariance",trainingSetup.block_5_project_BN.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block_5_add")
    convolution2dLayer([1 1],192,"Name","block_6_expand","Padding","same","Bias",trainingSetup.block_6_expand.Bias,"Weights",trainingSetup.block_6_expand.Weights)
    batchNormalizationLayer("Name","block_6_expand_BN","Epsilon",0.001,"Offset",trainingSetup.block_6_expand_BN.Offset,"Scale",trainingSetup.block_6_expand_BN.Scale,"TrainedMean",trainingSetup.block_6_expand_BN.TrainedMean,"TrainedVariance",trainingSetup.block_6_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_6_expand_relu")
    groupedConvolution2dLayer([3 3],1,192,"Name","block_6_depthwise","Padding","same","Stride",[2 2],"Bias",trainingSetup.block_6_depthwise.Bias,"Weights",trainingSetup.block_6_depthwise.Weights)
    batchNormalizationLayer("Name","block_6_depthwise_BN","Epsilon",0.001,"Offset",trainingSetup.block_6_depthwise_BN.Offset,"Scale",trainingSetup.block_6_depthwise_BN.Scale,"TrainedMean",trainingSetup.block_6_depthwise_BN.TrainedMean,"TrainedVariance",trainingSetup.block_6_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_6_depthwise_relu")
    convolution2dLayer([1 1],64,"Name","block_6_project","Padding","same","Bias",trainingSetup.block_6_project.Bias,"Weights",trainingSetup.block_6_project.Weights)
    batchNormalizationLayer("Name","block_6_project_BN","Epsilon",0.001,"Offset",trainingSetup.block_6_project_BN.Offset,"Scale",trainingSetup.block_6_project_BN.Scale,"TrainedMean",trainingSetup.block_6_project_BN.TrainedMean,"TrainedVariance",trainingSetup.block_6_project_BN.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],384,"Name","block_7_expand","Padding","same","Bias",trainingSetup.block_7_expand.Bias,"Weights",trainingSetup.block_7_expand.Weights)
    batchNormalizationLayer("Name","block_7_expand_BN","Epsilon",0.001,"Offset",trainingSetup.block_7_expand_BN.Offset,"Scale",trainingSetup.block_7_expand_BN.Scale,"TrainedMean",trainingSetup.block_7_expand_BN.TrainedMean,"TrainedVariance",trainingSetup.block_7_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_7_expand_relu")
    groupedConvolution2dLayer([3 3],1,384,"Name","block_7_depthwise","Padding","same","Bias",trainingSetup.block_7_depthwise.Bias,"Weights",trainingSetup.block_7_depthwise.Weights)
    batchNormalizationLayer("Name","block_7_depthwise_BN","Epsilon",0.001,"Offset",trainingSetup.block_7_depthwise_BN.Offset,"Scale",trainingSetup.block_7_depthwise_BN.Scale,"TrainedMean",trainingSetup.block_7_depthwise_BN.TrainedMean,"TrainedVariance",trainingSetup.block_7_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_7_depthwise_relu")
    convolution2dLayer([1 1],64,"Name","block_7_project","Padding","same","Bias",trainingSetup.block_7_project.Bias,"Weights",trainingSetup.block_7_project.Weights)
    batchNormalizationLayer("Name","block_7_project_BN","Epsilon",0.001,"Offset",trainingSetup.block_7_project_BN.Offset,"Scale",trainingSetup.block_7_project_BN.Scale,"TrainedMean",trainingSetup.block_7_project_BN.TrainedMean,"TrainedVariance",trainingSetup.block_7_project_BN.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","block_7_add");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],384,"Name","block_8_expand","Padding","same","Bias",trainingSetup.block_8_expand.Bias,"Weights",trainingSetup.block_8_expand.Weights)
    batchNormalizationLayer("Name","block_8_expand_BN","Epsilon",0.001,"Offset",trainingSetup.block_8_expand_BN.Offset,"Scale",trainingSetup.block_8_expand_BN.Scale,"TrainedMean",trainingSetup.block_8_expand_BN.TrainedMean,"TrainedVariance",trainingSetup.block_8_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_8_expand_relu")
    groupedConvolution2dLayer([3 3],1,384,"Name","block_8_depthwise","Padding","same","Bias",trainingSetup.block_8_depthwise.Bias,"Weights",trainingSetup.block_8_depthwise.Weights)
    batchNormalizationLayer("Name","block_8_depthwise_BN","Epsilon",0.001,"Offset",trainingSetup.block_8_depthwise_BN.Offset,"Scale",trainingSetup.block_8_depthwise_BN.Scale,"TrainedMean",trainingSetup.block_8_depthwise_BN.TrainedMean,"TrainedVariance",trainingSetup.block_8_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_8_depthwise_relu")
    convolution2dLayer([1 1],64,"Name","block_8_project","Padding","same","Bias",trainingSetup.block_8_project.Bias,"Weights",trainingSetup.block_8_project.Weights)
    batchNormalizationLayer("Name","block_8_project_BN","Epsilon",0.001,"Offset",trainingSetup.block_8_project_BN.Offset,"Scale",trainingSetup.block_8_project_BN.Scale,"TrainedMean",trainingSetup.block_8_project_BN.TrainedMean,"TrainedVariance",trainingSetup.block_8_project_BN.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","block_8_add");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],384,"Name","block_9_expand","Padding","same","Bias",trainingSetup.block_9_expand.Bias,"Weights",trainingSetup.block_9_expand.Weights)
    batchNormalizationLayer("Name","block_9_expand_BN","Epsilon",0.001,"Offset",trainingSetup.block_9_expand_BN.Offset,"Scale",trainingSetup.block_9_expand_BN.Scale,"TrainedMean",trainingSetup.block_9_expand_BN.TrainedMean,"TrainedVariance",trainingSetup.block_9_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_9_expand_relu")
    groupedConvolution2dLayer([3 3],1,384,"Name","block_9_depthwise","Padding","same","Bias",trainingSetup.block_9_depthwise.Bias,"Weights",trainingSetup.block_9_depthwise.Weights)
    batchNormalizationLayer("Name","block_9_depthwise_BN","Epsilon",0.001,"Offset",trainingSetup.block_9_depthwise_BN.Offset,"Scale",trainingSetup.block_9_depthwise_BN.Scale,"TrainedMean",trainingSetup.block_9_depthwise_BN.TrainedMean,"TrainedVariance",trainingSetup.block_9_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_9_depthwise_relu")
    convolution2dLayer([1 1],64,"Name","block_9_project","Padding","same","Bias",trainingSetup.block_9_project.Bias,"Weights",trainingSetup.block_9_project.Weights)
    batchNormalizationLayer("Name","block_9_project_BN","Epsilon",0.001,"Offset",trainingSetup.block_9_project_BN.Offset,"Scale",trainingSetup.block_9_project_BN.Scale,"TrainedMean",trainingSetup.block_9_project_BN.TrainedMean,"TrainedVariance",trainingSetup.block_9_project_BN.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block_9_add")
    convolution2dLayer([1 1],384,"Name","block_10_expand","Padding","same","Bias",trainingSetup.block_10_expand.Bias,"Weights",trainingSetup.block_10_expand.Weights)
    batchNormalizationLayer("Name","block_10_expand_BN","Epsilon",0.001,"Offset",trainingSetup.block_10_expand_BN.Offset,"Scale",trainingSetup.block_10_expand_BN.Scale,"TrainedMean",trainingSetup.block_10_expand_BN.TrainedMean,"TrainedVariance",trainingSetup.block_10_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_10_expand_relu")
    groupedConvolution2dLayer([3 3],1,384,"Name","block_10_depthwise","Padding","same","Bias",trainingSetup.block_10_depthwise.Bias,"Weights",trainingSetup.block_10_depthwise.Weights)
    batchNormalizationLayer("Name","block_10_depthwise_BN","Epsilon",0.001,"Offset",trainingSetup.block_10_depthwise_BN.Offset,"Scale",trainingSetup.block_10_depthwise_BN.Scale,"TrainedMean",trainingSetup.block_10_depthwise_BN.TrainedMean,"TrainedVariance",trainingSetup.block_10_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_10_depthwise_relu")
    convolution2dLayer([1 1],96,"Name","block_10_project","Padding","same","Bias",trainingSetup.block_10_project.Bias,"Weights",trainingSetup.block_10_project.Weights)
    batchNormalizationLayer("Name","block_10_project_BN","Epsilon",0.001,"Offset",trainingSetup.block_10_project_BN.Offset,"Scale",trainingSetup.block_10_project_BN.Scale,"TrainedMean",trainingSetup.block_10_project_BN.TrainedMean,"TrainedVariance",trainingSetup.block_10_project_BN.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],576,"Name","block_11_expand","Padding","same","Bias",trainingSetup.block_11_expand.Bias,"Weights",trainingSetup.block_11_expand.Weights)
    batchNormalizationLayer("Name","block_11_expand_BN","Epsilon",0.001,"Offset",trainingSetup.block_11_expand_BN.Offset,"Scale",trainingSetup.block_11_expand_BN.Scale,"TrainedMean",trainingSetup.block_11_expand_BN.TrainedMean,"TrainedVariance",trainingSetup.block_11_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_11_expand_relu")
    groupedConvolution2dLayer([3 3],1,576,"Name","block_11_depthwise","Padding","same","Bias",trainingSetup.block_11_depthwise.Bias,"Weights",trainingSetup.block_11_depthwise.Weights)
    batchNormalizationLayer("Name","block_11_depthwise_BN","Epsilon",0.001,"Offset",trainingSetup.block_11_depthwise_BN.Offset,"Scale",trainingSetup.block_11_depthwise_BN.Scale,"TrainedMean",trainingSetup.block_11_depthwise_BN.TrainedMean,"TrainedVariance",trainingSetup.block_11_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_11_depthwise_relu")
    convolution2dLayer([1 1],96,"Name","block_11_project","Padding","same","Bias",trainingSetup.block_11_project.Bias,"Weights",trainingSetup.block_11_project.Weights)
    batchNormalizationLayer("Name","block_11_project_BN","Epsilon",0.001,"Offset",trainingSetup.block_11_project_BN.Offset,"Scale",trainingSetup.block_11_project_BN.Scale,"TrainedMean",trainingSetup.block_11_project_BN.TrainedMean,"TrainedVariance",trainingSetup.block_11_project_BN.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","block_11_add");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],576,"Name","block_12_expand","Padding","same","Bias",trainingSetup.block_12_expand.Bias,"Weights",trainingSetup.block_12_expand.Weights)
    batchNormalizationLayer("Name","block_12_expand_BN","Epsilon",0.001,"Offset",trainingSetup.block_12_expand_BN.Offset,"Scale",trainingSetup.block_12_expand_BN.Scale,"TrainedMean",trainingSetup.block_12_expand_BN.TrainedMean,"TrainedVariance",trainingSetup.block_12_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_12_expand_relu")
    groupedConvolution2dLayer([3 3],1,576,"Name","block_12_depthwise","Padding","same","Bias",trainingSetup.block_12_depthwise.Bias,"Weights",trainingSetup.block_12_depthwise.Weights)
    batchNormalizationLayer("Name","block_12_depthwise_BN","Epsilon",0.001,"Offset",trainingSetup.block_12_depthwise_BN.Offset,"Scale",trainingSetup.block_12_depthwise_BN.Scale,"TrainedMean",trainingSetup.block_12_depthwise_BN.TrainedMean,"TrainedVariance",trainingSetup.block_12_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_12_depthwise_relu")
    convolution2dLayer([1 1],96,"Name","block_12_project","Padding","same","Bias",trainingSetup.block_12_project.Bias,"Weights",trainingSetup.block_12_project.Weights)
    batchNormalizationLayer("Name","block_12_project_BN","Epsilon",0.001,"Offset",trainingSetup.block_12_project_BN.Offset,"Scale",trainingSetup.block_12_project_BN.Scale,"TrainedMean",trainingSetup.block_12_project_BN.TrainedMean,"TrainedVariance",trainingSetup.block_12_project_BN.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block_12_add")
    convolution2dLayer([1 1],576,"Name","block_13_expand","Padding","same","Bias",trainingSetup.block_13_expand.Bias,"Weights",trainingSetup.block_13_expand.Weights)
    batchNormalizationLayer("Name","block_13_expand_BN","Epsilon",0.001,"Offset",trainingSetup.block_13_expand_BN.Offset,"Scale",trainingSetup.block_13_expand_BN.Scale,"TrainedMean",trainingSetup.block_13_expand_BN.TrainedMean,"TrainedVariance",trainingSetup.block_13_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_13_expand_relu")
    groupedConvolution2dLayer([3 3],1,576,"Name","block_13_depthwise","Padding","same","Stride",[2 2],"Bias",trainingSetup.block_13_depthwise.Bias,"Weights",trainingSetup.block_13_depthwise.Weights)
    batchNormalizationLayer("Name","block_13_depthwise_BN","Epsilon",0.001,"Offset",trainingSetup.block_13_depthwise_BN.Offset,"Scale",trainingSetup.block_13_depthwise_BN.Scale,"TrainedMean",trainingSetup.block_13_depthwise_BN.TrainedMean,"TrainedVariance",trainingSetup.block_13_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_13_depthwise_relu")
    convolution2dLayer([1 1],160,"Name","block_13_project","Padding","same","Bias",trainingSetup.block_13_project.Bias,"Weights",trainingSetup.block_13_project.Weights)
    batchNormalizationLayer("Name","block_13_project_BN","Epsilon",0.001,"Offset",trainingSetup.block_13_project_BN.Offset,"Scale",trainingSetup.block_13_project_BN.Scale,"TrainedMean",trainingSetup.block_13_project_BN.TrainedMean,"TrainedVariance",trainingSetup.block_13_project_BN.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],960,"Name","block_14_expand","Padding","same","Bias",trainingSetup.block_14_expand.Bias,"Weights",trainingSetup.block_14_expand.Weights)
    batchNormalizationLayer("Name","block_14_expand_BN","Epsilon",0.001,"Offset",trainingSetup.block_14_expand_BN.Offset,"Scale",trainingSetup.block_14_expand_BN.Scale,"TrainedMean",trainingSetup.block_14_expand_BN.TrainedMean,"TrainedVariance",trainingSetup.block_14_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_14_expand_relu")
    groupedConvolution2dLayer([3 3],1,960,"Name","block_14_depthwise","Padding","same","Bias",trainingSetup.block_14_depthwise.Bias,"Weights",trainingSetup.block_14_depthwise.Weights)
    batchNormalizationLayer("Name","block_14_depthwise_BN","Epsilon",0.001,"Offset",trainingSetup.block_14_depthwise_BN.Offset,"Scale",trainingSetup.block_14_depthwise_BN.Scale,"TrainedMean",trainingSetup.block_14_depthwise_BN.TrainedMean,"TrainedVariance",trainingSetup.block_14_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_14_depthwise_relu")
    convolution2dLayer([1 1],160,"Name","block_14_project","Padding","same","Bias",trainingSetup.block_14_project.Bias,"Weights",trainingSetup.block_14_project.Weights)
    batchNormalizationLayer("Name","block_14_project_BN","Epsilon",0.001,"Offset",trainingSetup.block_14_project_BN.Offset,"Scale",trainingSetup.block_14_project_BN.Scale,"TrainedMean",trainingSetup.block_14_project_BN.TrainedMean,"TrainedVariance",trainingSetup.block_14_project_BN.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","block_14_add");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],960,"Name","block_15_expand","Padding","same","Bias",trainingSetup.block_15_expand.Bias,"Weights",trainingSetup.block_15_expand.Weights)
    batchNormalizationLayer("Name","block_15_expand_BN","Epsilon",0.001,"Offset",trainingSetup.block_15_expand_BN.Offset,"Scale",trainingSetup.block_15_expand_BN.Scale,"TrainedMean",trainingSetup.block_15_expand_BN.TrainedMean,"TrainedVariance",trainingSetup.block_15_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_15_expand_relu")
    groupedConvolution2dLayer([3 3],1,960,"Name","block_15_depthwise","Padding","same","Bias",trainingSetup.block_15_depthwise.Bias,"Weights",trainingSetup.block_15_depthwise.Weights)
    batchNormalizationLayer("Name","block_15_depthwise_BN","Epsilon",0.001,"Offset",trainingSetup.block_15_depthwise_BN.Offset,"Scale",trainingSetup.block_15_depthwise_BN.Scale,"TrainedMean",trainingSetup.block_15_depthwise_BN.TrainedMean,"TrainedVariance",trainingSetup.block_15_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_15_depthwise_relu")
    convolution2dLayer([1 1],160,"Name","block_15_project","Padding","same","Bias",trainingSetup.block_15_project.Bias,"Weights",trainingSetup.block_15_project.Weights)
    batchNormalizationLayer("Name","block_15_project_BN","Epsilon",0.001,"Offset",trainingSetup.block_15_project_BN.Offset,"Scale",trainingSetup.block_15_project_BN.Scale,"TrainedMean",trainingSetup.block_15_project_BN.TrainedMean,"TrainedVariance",trainingSetup.block_15_project_BN.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block_15_add")
    convolution2dLayer([1 1],960,"Name","block_16_expand","Padding","same","Bias",trainingSetup.block_16_expand.Bias,"Weights",trainingSetup.block_16_expand.Weights)
    batchNormalizationLayer("Name","block_16_expand_BN","Epsilon",0.001,"Offset",trainingSetup.block_16_expand_BN.Offset,"Scale",trainingSetup.block_16_expand_BN.Scale,"TrainedMean",trainingSetup.block_16_expand_BN.TrainedMean,"TrainedVariance",trainingSetup.block_16_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_16_expand_relu")
    groupedConvolution2dLayer([3 3],1,960,"Name","block_16_depthwise","Padding","same","Bias",trainingSetup.block_16_depthwise.Bias,"Weights",trainingSetup.block_16_depthwise.Weights)
    batchNormalizationLayer("Name","block_16_depthwise_BN","Epsilon",0.001,"Offset",trainingSetup.block_16_depthwise_BN.Offset,"Scale",trainingSetup.block_16_depthwise_BN.Scale,"TrainedMean",trainingSetup.block_16_depthwise_BN.TrainedMean,"TrainedVariance",trainingSetup.block_16_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_16_depthwise_relu")
    convolution2dLayer([1 1],320,"Name","block_16_project","Padding","same","Bias",trainingSetup.block_16_project.Bias,"Weights",trainingSetup.block_16_project.Weights)
    batchNormalizationLayer("Name","block_16_project_BN","Epsilon",0.001,"Offset",trainingSetup.block_16_project_BN.Offset,"Scale",trainingSetup.block_16_project_BN.Scale,"TrainedMean",trainingSetup.block_16_project_BN.TrainedMean,"TrainedVariance",trainingSetup.block_16_project_BN.TrainedVariance)
    convolution2dLayer([1 1],1280,"Name","Conv_1","Bias",trainingSetup.Conv_1.Bias,"Weights",trainingSetup.Conv_1.Weights)
    batchNormalizationLayer("Name","Conv_1_bn","Epsilon",0.001,"Offset",trainingSetup.Conv_1_bn.Offset,"Scale",trainingSetup.Conv_1_bn.Scale,"TrainedMean",trainingSetup.Conv_1_bn.TrainedMean,"TrainedVariance",trainingSetup.Conv_1_bn.TrainedVariance)
    clippedReluLayer(6,"Name","out_relu")
    globalAveragePooling2dLayer("Name","global_average_pooling2d_1")
    fullyConnectedLayer(5,"Name","fc")
    softmaxLayer("Name","Logits_softmax")
    classificationLayer("Name","classoutput")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;
%% Connect Layer Branches
% Connect all the branches of the network to create the network graph.

lgraph = connectLayers(lgraph,"block_1_project_BN","block_2_expand");
lgraph = connectLayers(lgraph,"block_1_project_BN","block_2_add/in2");
lgraph = connectLayers(lgraph,"block_2_project_BN","block_2_add/in1");
lgraph = connectLayers(lgraph,"block_3_project_BN","block_4_expand");
lgraph = connectLayers(lgraph,"block_3_project_BN","block_4_add/in2");
lgraph = connectLayers(lgraph,"block_4_project_BN","block_4_add/in1");
lgraph = connectLayers(lgraph,"block_4_add","block_5_expand");
lgraph = connectLayers(lgraph,"block_4_add","block_5_add/in2");
lgraph = connectLayers(lgraph,"block_5_project_BN","block_5_add/in1");
lgraph = connectLayers(lgraph,"block_6_project_BN","block_7_expand");
lgraph = connectLayers(lgraph,"block_6_project_BN","block_7_add/in2");
lgraph = connectLayers(lgraph,"block_7_project_BN","block_7_add/in1");
lgraph = connectLayers(lgraph,"block_7_add","block_8_expand");
lgraph = connectLayers(lgraph,"block_7_add","block_8_add/in2");
lgraph = connectLayers(lgraph,"block_8_project_BN","block_8_add/in1");
lgraph = connectLayers(lgraph,"block_8_add","block_9_expand");
lgraph = connectLayers(lgraph,"block_8_add","block_9_add/in2");
lgraph = connectLayers(lgraph,"block_9_project_BN","block_9_add/in1");
lgraph = connectLayers(lgraph,"block_10_project_BN","block_11_expand");
lgraph = connectLayers(lgraph,"block_10_project_BN","block_11_add/in2");
lgraph = connectLayers(lgraph,"block_11_project_BN","block_11_add/in1");
lgraph = connectLayers(lgraph,"block_11_add","block_12_expand");
lgraph = connectLayers(lgraph,"block_11_add","block_12_add/in2");
lgraph = connectLayers(lgraph,"block_12_project_BN","block_12_add/in1");
lgraph = connectLayers(lgraph,"block_13_project_BN","block_14_expand");
lgraph = connectLayers(lgraph,"block_13_project_BN","block_14_add/in2");
lgraph = connectLayers(lgraph,"block_14_project_BN","block_14_add/in1");
lgraph = connectLayers(lgraph,"block_14_add","block_15_expand");
lgraph = connectLayers(lgraph,"block_14_add","block_15_add/in2");
lgraph = connectLayers(lgraph,"block_15_project_BN","block_15_add/in1");

%% Train mobilenetv2
% Train the network using the specified options and training data.

[net, traininfo] = trainNetwork(augimdsTrain,lgraph,opts);

%% Save trained model
mobilenet_alldata_aug = net;

save mobilenet_alldata_aug