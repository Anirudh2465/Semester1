% Clear workspace, close figures, and clear command window
clear all
close all
clc

% Start stopwatch timer
tic

% Display information
fprintf('Classification using pseudoInverse for Network Anomaly Detection\n');
fprintf('Data read: starting\n');

% Load training data and labels
A1 = csvread("Training_data.csv");    % Training data (features)
B1 = csvread("Training_labels.csv");  % Training labels (binary classes: 0 or 1)

% Load testing data and labels
A2 = csvread("Testing_data.csv");     % Testing data
B2 = csvread("Testing_labels.csv");   % Testing labels

% Stop stopwatch and display elapsed time for reading data
toc
fprintf('Data read: completed\n');

% Create binary representation of training and testing labels
Btrain = [B1, ones(size(B1)) - B1];
Btest = [B2, ones(size(B2)) - B2];

% Reduce features (adjust as needed)
% For instance, removing the 27th column (assuming it's dependent)
Ar1 = A1(:, [1:26, 28:end]);
Ar2 = A2(:, [1:26, 28:end]);

% Compute pseudo-inverse
AtA = Ar1' * Ar1;
psinA = inv(AtA) * Ar1';

% Compute weights
weights = psinA * Btrain;

% Make predictions on the test set
pred = Ar2 * weights;

% Convert predictions to binary classes based on a threshold (e.g., 0.5)
pred_labels = (pred(:, 1) < pred(:, 2));

B_comp = (Btest(:, 1) < Btest(:, 2));

% Evaluate accuracy
accuracy = sum(pred_labels == B_comp) / length(B_comp) * 100;

% Display accuracy
fprintf('Accuracy of the prediction: %.2f%%\n',accuracy);