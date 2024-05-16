%%% Compare the performances of the trained models
%% Import trained models

alexnet_all = load("alexnet_alldata_aug.mat");
alexnet = alexnet_all.modelAug;
%%
mobilenetv2_all = load("mobilenet_alldata_aug.mat");
mobilenetv2 = mobilenetv2_all.net;
%%
resnet50_all = load("resnet_alldata_aug.mat");
resnet50 = resnet50_all.net;
%% AlexNet performance
%% 
% Get all validation images and resize to the appropriate input size

alexnet_valimds = imageDatastore(alexnet_all.augimdsValidation.Files, 'LabelSource', 'foldernames');

inputSize = [227 227 3];
alexnet_valimds = augmentedImageDatastore(inputSize, alexnet_valimds);
%% 
% Classify images with the trained model

[alexnet_valpred, ~] = classify(alexnet, alexnet_valimds, 'MiniBatchSize', 10);
%%
alexnet_splitfiles = split(alexnet_valimds.Files, '\');
alexnet_valimdslabels = categorical(alexnet_splitfiles(:, 9));
%%
[cmAug, orderAug] = confusionmat(alexnet_valpred, alexnet_valimdslabels);
%%
confusionchart(cmAug, orderAug)
%%
% Extract true positives (TP) for each class
TP = diag(cmAug);

% Sum up all actual positives for each class (TP + FN)
actual_positives = sum(cmAug, 2);

% Compute sensitivity (TP / (TP + FN)) for each class
sensitivity = TP ./ actual_positives;

% If any actual positives are zero, set sensitivity to NaN to avoid division by zero
sensitivity(actual_positives == 0) = NaN;

% Display or use sensitivity for each class as needed
disp('Sensitivity (True Positive Rate) for each class:');
disp(sensitivity);
%%
% Initialize an array to store specificities for each class
specificities = zeros(5, 1);

for i = 1:5
    % Extract true negatives (TN) for the current class
    TN = sum(sum(cmAug)) - sum(cmAug(i,:)) - sum(cmAug(:,i)) + cmAug(i,i);

    % Sum up all actual negatives (TN + FP) for the current class
    actual_negatives = sum(sum(cmAug)) - sum(cmAug(i,:));

    % Compute specificity (TN / (TN + FP)) for the current class
    specificities(i) = TN / actual_negatives;

    % If any actual negatives are zero, set specificity to NaN to avoid division by zero
    if actual_negatives == 0
        specificities(i) = NaN;
    end
end

% Display or use specificity for each class as needed
disp('Specificity for each class:');
disp(specificities);
%%
% Extract the diagonal elements of the confusion matrix (true positives for each class)
TP = diag(cmAug);

% Compute the total number of correctly classified samples (sum of true positives)
correctly_classified = sum(TP);

% Compute the total number of samples (sum of all elements in the confusion matrix)
total_samples = sum(cmAug(:));

% Compute accuracy (correctly classified samples / total samples)
accuracy = correctly_classified / total_samples;

disp(['Accuracy: ', num2str(accuracy)]);
%% MobileNetv2 performance
% Get all validation images and resize to the appropriate input size

mobilenet_valimds = imageDatastore(mobilenetv2_all.augimdsValidation.Files, 'LabelSource', 'foldernames');

inputSize = [224 224 3];
mobilenet_valimds = augmentedImageDatastore(inputSize, mobilenet_valimds, 'ColorPreprocessing', 'gray2rgb');
%% 
% Classify images with the trained model

[mobilenet_valpred, ~] = classify(mobilenetv2, mobilenet_valimds, 'MiniBatchSize', 10);
%%
mobilenet_splitfiles = split(mobilenet_valimds.Files, '\');
mobilenet_valimdslabels = categorical(mobilenet_splitfiles(:, 9));
%%
[cmAug, orderAug] = confusionmat(mobilenet_valpred, mobilenet_valimdslabels);
confusionchart(cmAug, orderAug)
%%
% Extract true positives (TP) for each class
TP = diag(cmAug);

% Sum up all actual positives for each class (TP + FN)
actual_positives = sum(cmAug, 2);

% Compute sensitivity (TP / (TP + FN)) for each class
sensitivity = TP ./ actual_positives;

% If any actual positives are zero, set sensitivity to NaN to avoid division by zero
sensitivity(actual_positives == 0) = NaN;

% Display or use sensitivity for each class as needed
disp('Sensitivity (True Positive Rate) for each class:');
disp(sensitivity);
%%
% Initialize an array to store specificities for each class
specificities = zeros(5, 1);

for i = 1:5
    % Extract true negatives (TN) for the current class
    TN = sum(sum(cmAug)) - sum(cmAug(i,:)) - sum(cmAug(:,i)) + cmAug(i,i);

    % Sum up all actual negatives (TN + FP) for the current class
    actual_negatives = sum(sum(cmAug)) - sum(cmAug(i,:));

    % Compute specificity (TN / (TN + FP)) for the current class
    specificities(i) = TN / actual_negatives;

    % If any actual negatives are zero, set specificity to NaN to avoid division by zero
    if actual_negatives == 0
        specificities(i) = NaN;
    end
end

% Display or use specificity for each class as needed
disp('Specificity for each class:');
disp(specificities);
%%
% Extract the diagonal elements of the confusion matrix (true positives for each class)
TP = diag(cmAug);

% Compute the total number of correctly classified samples (sum of true positives)
correctly_classified = sum(TP);

% Compute the total number of samples (sum of all elements in the confusion matrix)
total_samples = sum(cmAug(:));

% Compute accuracy (correctly classified samples / total samples)
accuracy = correctly_classified / total_samples;

disp(['Accuracy: ', num2str(accuracy)]);
%% ResNet50 performance
% Get all validation images and resize to the appropriate input size

resnet_valimds = imageDatastore(resnet50_all.augimdsValidation.Files, 'LabelSource', 'foldernames');

inputSize = [224 224 3];
resnet_valimds = augmentedImageDatastore(inputSize, resnet_valimds, 'ColorPreprocessing', 'gray2rgb');
%% 
% Classify images with the trained model

[resnet_valpred, scores] = classify(resnet50, resnet_valimds, 'MiniBatchSize', 10);
%%
resnet_splitfiles = split(resnet_valimds.Files, '\');
resnet_valimdslabels = categorical(resnet_splitfiles(:, 9));
%%
[cmAug, orderAug] = confusionmat(resnet_valpred, resnet_valimdslabels);
confusionchart(cmAug, orderAug)
%%
% Extract true positives (TP) for each class
TP = diag(cmAug);

% Sum up all actual positives for each class (TP + FN)
actual_positives = sum(cmAug, 2);

% Compute sensitivity (TP / (TP + FN)) for each class
sensitivity = TP ./ actual_positives;

% If any actual positives are zero, set sensitivity to NaN to avoid division by zero
sensitivity(actual_positives == 0) = NaN;

% Display or use sensitivity for each class as needed
disp('Sensitivity (True Positive Rate) for each class:');
disp(sensitivity);
%%
% Initialize an array to store specificities for each class
specificities = zeros(5, 1);

for i = 1:5
    % Extract true negatives (TN) for the current class
    TN = sum(sum(cmAug)) - sum(cmAug(i,:)) - sum(cmAug(:,i)) + cmAug(i,i);

    % Sum up all actual negatives (TN + FP) for the current class
    actual_negatives = sum(sum(cmAug)) - sum(cmAug(i,:));

    % Compute specificity (TN / (TN + FP)) for the current class
    specificities(i) = TN / actual_negatives;

    % If any actual negatives are zero, set specificity to NaN to avoid division by zero
    if actual_negatives == 0
        specificities(i) = NaN;
    end
end

% Display or use specificity for each class as needed
disp('Specificity for each class:');
disp(specificities);
%%
% Extract the diagonal elements of the confusion matrix (true positives for each class)
TP = diag(cmAug);

% Compute the total number of correctly classified samples (sum of true positives)
correctly_classified = sum(TP);

% Compute the total number of samples (sum of all elements in the confusion matrix)
total_samples = sum(cmAug(:));

% Compute accuracy (correctly classified samples / total samples)
accuracy = correctly_classified / total_samples;

disp(['Accuracy: ', num2str(accuracy)]);
%% Overall accuracy/sensitivity/specificity comparisons across models

% Compute confusion matrices for each model
[alexnet_cm, order] = confusionmat(alexnet_valpred, alexnet_valimdslabels);
[mobilenet_cm, ~] = confusionmat(mobilenet_valpred, mobilenet_valimdslabels);
[resnet_cm, ~] = confusionmat(resnet_valpred, resnet_valimdslabels);

figure('Position', [520 441 848 438])

% Initialize array to store overall comparisons
overall_comparisons = zeros(3, 3);

% Loop through each model
for i = 1:3
    % Select the confusion matrix for the current model
    if i == 1
        temp_cm = alexnet_cm;
        model_name = 'AlexNet';
    elseif i == 2
        temp_cm = mobilenet_cm;
        model_name = 'MobileNetv2';
    else
        temp_cm = resnet_cm;
        model_name = 'ResNet50';
    end

    % Compute metrics for the current model
    [accuracy, sensitivity, specificity] = get_metrics(temp_cm);
    
    % Store metrics in overall comparisons array
    overall_comparisons(i, :) = [accuracy, mean(sensitivity), mean(specificity)];
end

overall_comp_inv = overall_comparisons';

% Plot bar chart for overall comparisons
hb = bar(overall_comp_inv, 1, 'GroupWidth', 0.9, 'EdgeColor', 'none');

display(get(hb(1)))

% Add y-values over each bar
for i = 1:size(overall_comp_inv, 1)
    for j = 1:size(overall_comp_inv, 2)
        x = hb(i).XData+hb(j).XOffset;
        text(x(i), overall_comp_inv(i, j)/2, [num2str(round(overall_comp_inv(i, j), 2)*100), '%'], ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'Color', 'white');
    end
end

color_idx = [[0.137 0.329 0.929]; [0.055 0.722  0.165]; [0.741 0.294 0.949]];
for k = 1:size(overall_comp_inv, 1)
    hb(k).FaceColor = color_idx(k, :);
end

legend('AlexNet','MobileNetv2', 'ResNet50', 'Location', 'eastoutside')
set(gca, 'XTickLabel', {'Accuracy', 'Sensitivity', 'Specificity'}, 'YTick', {}, 'FontSize', 10)
%% Sensitivity/Specificity for each class

% Initialize array to store overall comparisons
overall_sensitivies = zeros(3, 5);
overall_specificities = zeros(3, 5);
% Loop through each model
for i = 1:3
    % Select the confusion matrix for the current model
    if i == 1
        temp_cm = alexnet_cm;
        model_name = 'AlexNet';
    elseif i == 2
        temp_cm = mobilenet_cm;
        model_name = 'MobileNetv2';
    else
        temp_cm = resnet_cm;
        model_name = 'ResNet50';
    end

    % Compute metrics for the current model
    [accuracy, sensitivity, specificity] = get_metrics(temp_cm);
    
    % Store metrics in overall comparisons array
    overall_sensitivies(i, :) = sensitivity;
    overall_specificities(i, :) = specificity;
end
%%
% Plot bar chart for overall comparisons
figure('Position', [520 521 1082 358])
hb = bar(overall_sensitivies', 1, 'GroupWidth', 0.9);
xticklabels(order)

get(hb(1));

% Add y-values over each bar
for i = 1:size(overall_sensitivies, 1)
    for j = 1:size(overall_sensitivies, 2)
        x = hb(i).XData+hb(i).XOffset;
        text(x(j), overall_sensitivies(i, j)/2, [num2str(round(overall_sensitivies(i, j), 2)*100), '%'], ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'Color', 'white', ...
             'FontWeight', 'bold');
    end
end

color_idx = [[0.137 0.329 0.929]; [0.055 0.722  0.165]; [0.741 0.294 0.949]];
for k = 1:size(overall_comp_inv, 1)
    hb(k).FaceColor = color_idx(k, :);
end

legend('AlexNet','MobileNetv2', 'ResNet50', 'Location', 'eastoutside')
set(gca, 'YTick', {}, 'FontSize', 10)
%%
% Plot bar chart for overall comparisons
figure('Position', [520 521 1082 358])
hb = bar(overall_specificities', 1, 'GroupWidth', 0.9);
xticklabels(order)

get(hb(1));

% Add y-values over each bar
for i = 1:size(overall_specificities, 1)
    for j = 1:size(overall_specificities, 2)
        x = hb(i).XData+hb(i).XOffset;
        text(x(j), overall_specificities(i, j)/2, [num2str(round(overall_specificities(i, j), 2)*100), '%'], ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'Color', 'white', ...
             'FontWeight', 'bold');
    end
end

color_idx = [[0.137 0.329 0.929]; [0.055 0.722  0.165]; [0.741 0.294 0.949]];
for k = 1:size(overall_comp_inv, 1)
    hb(k).FaceColor = color_idx(k, :);
end

legend('AlexNet','MobileNetv2', 'ResNet50', 'Location', 'eastoutside')
set(gca, 'YTick', {}, 'FontSize', 10)
%% Classify new images

newimgs_path = 'C:\Users\kem00\OneDrive\Documents\forAndrew\project\UnseenImages';
test_imds = imageDatastore(newimgs_path);

% inputSize = [227 227 3];
% test_imds_aug = augmentedImageDatastore(inputSize, test_imds, 'ColorPreprocessing', 'gray2rgb');
% Predictions

inputSize = [227 227 3];
test_imds_aug = augmentedImageDatastore(inputSize, test_imds, 'ColorPreprocessing', 'gray2rgb');
[alexnet_testpred, alexnet_testscores] = classify(alexnet, test_imds_aug, 'MiniBatchSize', 10);

inputSize = [224 224 3];
test_imds_aug = augmentedImageDatastore(inputSize, test_imds, 'ColorPreprocessing', 'gray2rgb');
[mobilenet_testpred, mobilenet_testscores] = classify(mobilenetv2, test_imds_aug, 'MiniBatchSize', 10);

inputSize = [224 224 3];
test_imds_aug = augmentedImageDatastore(inputSize, test_imds, 'ColorPreprocessing', 'gray2rgb');
[resnet_testpred, resnet_testscores] = classify(resnet50, test_imds_aug, 'MiniBatchSize', 10);
%%
img_num = 2;
% Load an example image
image_data = imread(string(test_imds.Files(img_num)));

% Create a figure
figure;

% Create subplot for the image
subplot(1, 2, 1);
imshow(image_data);
axis off; % Turn off axis
title(''); % Remove title

% Generate sample data for the bar graph (replace this with your actual data)
% categories = {'Category 1', 'Category 2', 'Category 3', 'Category 4', 'Category 5'};
values = scores(img_num, :);

% Create subplot for the horizontal bar graph
subplot(1, 2, 2);
h = barh(values);
h.EdgeColor = 'none';
yticks(1:numel(order));
yticklabels(order);
xlabel('Values');
axis off; % Turn off axis
title(''); % Remove title

% Display values in the middle of each bar
for i = 1:numel(values)
    % Adjust the x-coordinate based on the value of the bar
    if values(i) < 0.12
        x = values(i)+0.06; % Move the label to the right for better visibility
        text(x, i, [num2str(round(values(i) * 100)), '%'], 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 12, 'FontWeight', 'normal');
        text(-0.7, i, order(i), 'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', 'FontSize', 12);
    else
        x = values(i) / 2;
        text(x, i, [num2str(round(values(i) * 100)), '%'], 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 12, 'Color', 'white', 'FontWeight', 'bold');
        text(-0.7, i, order(i), 'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', 'FontSize', 12);
    end
end

% Adjust the layout
set(gcf, 'Position', [100, 100, 1200, 400]);  % Set figure size

% Adjust the spacing between subplots
subplot(1, 2, 1);
set(gca, 'Position', [0.1, 0.1, 0.4, 0.8]);  % Adjust position of image subplot
subplot(1, 2, 2);
set(gca, 'Position', [0.402, 0.1, 0.3, 0.8]);  % Adjust position of bar graph subplot

% Add text labels to the left of the bar graph
% for i = 1:numel(order)
%     text(-0.1, i, order(i), 'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', 'FontSize', 12);
% end
%%
img_num = 2;
% Load an example image
image_data = imread(string(test_imds.Files(img_num)));

% Create a figure
figure;

% Create subplot for the image
subplot(1, 2, 1);
imshow(image_data);
axis off; % Turn off axis
title(''); % Remove title

% Generate sample data for the bar graph (replace this with your actual data)
% categories = {'Category 1', 'Category 2', 'Category 3', 'Category 4', 'Category 5'};
values = [alexnet_testscores(img_num, :); mobilenet_testscores(img_num, :); resnet_testscores(img_num, :)]';

% Create subplot for the horizontal bar graph
subplot(1, 2, 2);
h = barh(values, 'EdgeColor', 'none', 'BarWidth', 1);
yticks(1:numel(order));
yticklabels(order);
xlabel('Values');
axis off; % Turn off axis
title(''); % Remove title

% Remove black line for axis
ax = gca;
ax.Box = 'off'; % Remove the axis box

% Display values in the middle of each bar
for i = 1:length(values)
    % Adjust the x-coordinate based on the value of the bar
    if values(i) < 0.12
        x = values(i)+0.06; % Move the label to the right for better visibility
        text(x, i, [num2str(round(values(i) * 100)), '%'], 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 12, 'FontWeight', 'normal');
        text(-0.7, i, order(i), 'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', 'FontSize', 12);
    else
        x = values(i) / 2;
        text(x, i, [num2str(round(values(i) * 100)), '%'], 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 12, 'Color', 'white', 'FontWeight', 'bold');
        text(-0.7, i, order(i), 'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', 'FontSize', 12);
    end
end

% Adjust the layout
set(gcf, 'Position', [100, 100, 1200, 400]);  % Set figure size

% Adjust the spacing between subplots
subplot(1, 2, 1);
set(gca, 'Position', [0.1, 0.1, 0.4, 0.8]);  % Adjust position of image subplot
subplot(1, 2, 2);
set(gca, 'Position', [0.402, 0.1, 0.3, 0.8]);  % Adjust position of bar graph subplot
%%
img_num = 2;

% Load an example image
image_data = imread(string(test_imds.Files(img_num)));

% Create a figure
figure;

% Create subplot for the image
subplot(1, 2, 1);
imshow(image_data);
axis off; % Turn off axis
title(''); % Remove title

% Generate sample data for the bar graph (replace this with your actual data)
% Assuming 'order' is defined elsewhere
vals = [alexnet_testscores(img_num, :); mobilenet_testscores(img_num, :); resnet_testscores(img_num, :)]';

% Create subplot for the horizontal bar graph
subplot(1, 2, 2);
h = barh(vals, 'EdgeColor', 'none', 'BarWidth', 1, 'GroupWidth', 0.9);
yticks(1:numel(order));
yticklabels(order);
xlabel('Values');
axis off; % Turn off axis
title(''); % Remove title

% Remove black line for axis
ax = gca;
ax.Box = 'off'; % Remove the axis box


% Iterate through each bar in the category
for i = 1:size(vals, 2)
    for j = 1:size(vals, 1)
        if vals(j,i) < 0.12
            % Calculate the center position of the bar
            x = h(i).XData+h(i).XOffset;
            % Display the percent label in the middle of the bar
            text(vals(j, i)+0.05, x(j), [num2str(round(vals(j, i) * 100)), '%'], 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 12, 'FontWeight', 'normal');
%             text(-0.7, i, order(i), 'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', 'FontSize', 12);
        else
            x = (h(i).XData+h(i).XOffset);
            % Display the percent label in the middle of the bar
            text(vals(j, i)/2, x(j), [num2str(round(vals(j, i) * 100)), '%'], 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 12, 'Color', 'white', 'FontWeight', 'bold');
%             text(-0.7, i, order(i), 'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', 'FontSize', 12);
        end
    end
end

% Display the category label on the left side of the bar
text([-0.5 -0.5 -0.5 -0.5 -0.5], 1:size(vals,1), order, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', 'FontSize', 12);

% Adjust the layout
set(gcf, 'Position', [100, 100, 1200, 400]);  % Set figure size

% Adjust the spacing between subplots
subplot(1, 2, 1);
set(gca, 'Position', [0.1, 0.1, 0.4, 0.8]);  % Adjust position of image subplot
subplot(1, 2, 2);
set(gca, 'Position', [0.4005, 0.1, 0.4, 0.8]);  % Adjust position of bar graph subplot
%%
img_num = 15;
% Load an example image
image_data = imread(string(test_imds.Files(img_num)));

% Create a figure
figure;

% Create subplot for the image
imshow(image_data);
%%
img_num = 16;
vals = [alexnet_testscores(img_num, :); mobilenet_testscores(img_num, :); resnet_testscores(img_num, :)]';
% Plot bar chart for overall comparisons
figure('Position', [1272 166 647 408])
hb = bar(vals, 1, 'GroupWidth', 0.9);
xticklabels(order)

get(hb(1));

% Add y-values over each bar
for i = 1:size(vals, 2)
    for j = 1:size(vals, 1)
        if vals(j,i) < .11
            x = hb(i).XData+hb(i).XOffset;
            text(x(j), vals(j,i)/2+0.04, [num2str(round(vals(j,i), 2)*100), '%'], ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'Color', 'black', ...
                'FontWeight', 'bold');

        else
            x = hb(i).XData+hb(i).XOffset;
            text(x(j), vals(j,i)/2, [num2str(round(vals(j,i), 2)*100), '%'], ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'Color', 'white', ...
                'FontWeight', 'bold');
        end
        
    end
end

color_idx = [[0.137 0.329 0.929]; [0.055 0.722  0.165]; [0.741 0.294 0.949]];
for k = 1:size(overall_comp_inv, 1)
    hb(k).FaceColor = color_idx(k, :);
end

set(gca,'box','off')
set(gca,'ycolor','w','ytick',[])
% legend('AlexNet','MobileNetv2', 'ResNet50', 'Location', 'northeast')
%%
vals = [alexnet_testscores(img_num, :); mobilenet_testscores(img_num, :); resnet_testscores(img_num, :)]';
% Plot bar chart for overall comparisons
figure('Position', [520 521 1100 1082])
hb = barh(vals, 1, 'GroupWidth', 0.9);
% yticklabels(order)

get(hb(1));

% Add y-values nex to each bar
for i = 1:size(vals, 2)
    for j = 1:size(vals, 1)
        if vals(j,i) < 0.11
            x = hb(i).XData+hb(i).XOffset;
            text(vals(j,i)/2+0.08, x(1, j), [num2str(round(vals(j, i) * 100)), '%'], ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 12, 'FontWeight', 'normal');

        else
            x = hb(i).XData+hb(i).XOffset;
            text(vals(j,i)/2, x(1, j), [num2str(round(vals(j, i) * 100)), '%'], ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 12, 'Color', 'white', 'FontWeight', 'bold');

        end

    end
end

set(gca,'box','off')
set(gca,'xcolor','w','ycolor','w','xtick',[],'ytick',[])
exportgraphics(gcf, 'figex.png', 'ContentType', 'vector', 'BackgroundColor', 'none')
% legend('AlexNet','MobileNetv2', 'ResNet50', 'Location', 'northeast')
%% Functions

function [accuracy, sensitivity, specificity] = get_metrics(cm)
    
    %%% ACCURACY
    TP = diag(cm);

    % Compute the total number of correctly classified samples (sum of true positives)
    correctly_classified = sum(TP);
    
    % Compute the total number of samples (sum of all elements in the confusion matrix)
    total_samples = sum(cm(:));
    
    % Compute accuracy (correctly classified samples / total samples)
    accuracy = correctly_classified / total_samples;

    %%% SENSITIVITY
    % Extract true positives (TP) for each class
    TP = diag(cm);
    
    % Sum up all actual positives for each class (TP + FN)
    actual_positives = sum(cm, 2);
    
    % Compute sensitivity (TP / (TP + FN)) for each class
    sensitivity = TP ./ actual_positives;
    
    % If any actual positives are zero, set sensitivity to NaN to avoid division by zero
    sensitivity(actual_positives == 0) = NaN;

    %%% SPECIFICITY
    % Initialize an array to store specificities for each class
    specificity = zeros(5, 1);
    
    for i = 1:5
        % Extract true negatives (TN) for the current class
        TN = sum(sum(cm)) - sum(cm(i,:)) - sum(cm(:,i)) + cm(i,i);
    
        % Sum up all actual negatives (TN + FP) for the current class
        actual_negatives = sum(sum(cm)) - sum(cm(i,:));
    
        % Compute specificity (TN / (TN + FP)) for the current class
        specificity(i) = TN / actual_negatives;
    
        % If any actual negatives are zero, set specificity to NaN to avoid division by zero
        if actual_negatives == 0
            specificity(i) = NaN;
        end
    end

end