%% Load alexnet

model = alexnet;

%% Inspect the model
analyzeNetwork(model)

%% Load image data and build image datastore

dataFolder = 'C:\Users\...\project\TrashDataset';
categories = {'BeverageCans', 'BottleCaps', 'CigaretteButts', 'PlasticBags', 'PlasticBottlesJugs'};
imds = imageDatastore(fullfile(dataFolder, categories), 'LabelSource', 'foldernames');
tbl = countEachLabel(imds);
disp (tbl)

countEachLabel(imds)
%% Create a pie chart with the data distributions
labels = cellstr(countEachLabel(imds).Label)
counts = countEachLabel(imds).Count
percentages = counts / sum(counts) * 100;

% Creating the pie chart
hPie = pie(percentages, labels);

% Adding percentages as text annotations in the slices
for i = 1:numel(percentages)
    % Calculate text position
    pos = hPie(2*i).Position;
    % Shift text position further towards the center of the pie slice
    pos(1) = pos(1) + -0.35 * pos(1);
    pos(2) = pos(2) + -0.35 * pos(2);
    
    % Determine text color based on slice index
    if i == 5
        textColor = [0 0 0]; % Black text for yellow slice
    else
        textColor = [1 1 1]; % White text for other slices
    end
    
    text(pos(1), pos(2), sprintf('%.1f%%', percentages(i)), 'HorizontalAlignment', 'center', 'Color', textColor);
end
%% Pre-process images for AlexNet (resize images)
% AlexNet can only process RGB images that are 227-by-227

imds.ReadFcn = @(filename)readAndPreprocessImage(filename, inputSize);
%% Divide data into training (70%) and validation sets (30%)

[trainingSet, validationSet] = splitEachLabel(imds, 0.7, 'randomized');

countEachLabel(trainingSet)
countEachLabel(validationSet)
%% Perform transfer tearning
% Freeze all but last three layers, then configure own final three laters
% with the correct num of classes

layersTransfer = model.Layers(1:end-3);
numClasses = 5;

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];
%% Configure training options

options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',3e-5, ...
    'Shuffle','every-epoch', ...
    'ValidationData',validationSet, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% Retrain AlexNet

modelTransfer = trainNetwork(trainingSet,layers,options);

%% Save the model

save modelTransfer

%% Classify the validation images using the fine-tuned network

[YPred,scores] = classify(modelTransfer,validationSet, 'MiniBatchSize',10);

%% Calculate the classification accuracy on the validation set
% Accuracy is the fraction of labels that the network predicts correctly

YValidation = validationSet.Labels;

[cm, order] = confusionmat(YPred, YValidation);
confusionchart(cm, order)

cc = cm(1);
dc = cm(2);
cd = cm(3);
dd = cm(4);

accuracy = mean(YPred == YValidation)*100;
fprintf("The validation accuracy is: %.2f %%\n", accuracy);

% Sensitivity -- how often the model predicts TP
sensitivity = dd/(dc + dd +eps)*100;
fprintf("The sensitivity is: %.2f %%\n", sensitivity);

% Specificity -- how often the model predicts TN
specificity = cc/(cc + cd +eps)*100;
fprintf("The specificity is: %.2f %%\n", specificity);

%% Display example validation set images that have been incorrectly classified
index = [];

for i = 1:1:sum(countEachLabel(validationSet).Count)
    if YPred(i) ~= YValidation(i)
        index = [index; i];
    end
end

imagesIndex = subset(validationSet, index);

% display a maximum of 6 images
if length(index) < 6
    montage(imagesIndex)
    title("Incorrectly classified images");
else
    montage({preview(subset(imagesIndex, 1)), preview(subset(imagesIndex, 2)), preview(subset(imagesIndex, 3)), ...
    preview(subset(imagesIndex, 4)), preview(subset(imagesIndex, 5)), preview(subset(imagesIndex, 6))})
    title("Incorrectly classified images");
end

%% Train AlexNet with data augmentation
% Define image data augmenter that randomly scales, reflects, or translates
% images

pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
%% Building the augmented training and validation sets

inputSize = model.Layers(1).InputSize;
augimdsTrain = augmentedImageDatastore(inputSize(1:2),trainingSet, ...
    'DataAugmentation',imageAugmenter);

disp(augimdsTrain.NumObservations)

augimdsValidation = augmentedImageDatastore(inputSize(1:2),validationSet);

disp(augimdsValidation.NumObservations)
%% Train the network with augmented datasets

options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',3e-5, ...
    'Shuffle','every-epoch', ...
    'ValidationData',validationSet, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

modelAug = trainNetwork(augimdsTrain,layers,options);

%% Save trained model
save modelAug

%% Classify the validation images using the fine-tuned network

[YPredAug,probsAug] = classify(modelAug,augimdsValidation, 'MiniBatchSize', 10);

%% Calculate the classification accuracy on the validation set

YValidationAug = validationSet.Labels;

[cmAug, orderAug] = confusionmat(YPredAug, YValidationAug);
confusionchart(cmAug, orderAug)

ccAug = cmAug(1);
dcAug = cmAug(2);
cdAug = cmAug(3);
ddAug = cmAug(4);

accuracyAug = mean(YPredAug == YValidationAug)*100;
fprintf("The validation accuracy is: %.2f %%\n", accuracyAug);

sensitivityAug = ddAug/(dcAug + ddAug + eps)*100;
fprintf("The sensitivity is: %.2f %%\n", sensitivityAug);

specificityAug = ccAug/(ccAug + cdAug + eps)*100;
fprintf("The specificity is: %.2f %%\n", specificityAug);

%% Display example images
indexAug = [];

for i = 1:1:sum(countEachLabel(validationSet).Count)
    if YPredAug(i) ~= YValidationAug(i)
        indexAug = [indexAug; i];
    end
end

imagesIndexAug = subset(validationSet, indexAug);

if length(indexAug) < 6
    montage(imagesIndexAug)
    title("Incorrectly classified images");
else
    montage({preview(subset(imagesIndexAug, 1)), preview(subset(imagesIndexAug, 2)), preview(subset(imagesIndexAug, 3)), ...
    preview(subset(imagesIndex, 4)), preview(subset(imagesIndexAug, 5)), preview(subset(imagesIndexAug, 6))})
    title("Incorrectly classified images");
end

%% Functions
function Iout = readAndPreprocessImage(filename, inputSize)

I = imread(filename);

% Some images may be grayscale. Replicate the image 3 times to
% create an RGB image.
if ismatrix(I)
    I = cat(3,I,I,I);
end

% Resize the image as required for the CNN.
Iout = imresize(I, inputSize(1:2));

end