poolobj = gcp('nocreate');
if isempty(poolobj)
    parpool; 
end

filePath = "C:\Users\sahil\OneDrive\Desktop\NEW_DOE_dataset.xlsx"; 

data = readmatrix(filePath);
X = data(:, 1:5);          
Y_hardness = data(:, 6);     
Y_tensile = data(:, 7);    

Y = [Y_hardness, Y_tensile];
[N_runs, num_vars] = size(X);

disp('Training surrogate models mapping inputs to responses...');
mdl_hardness = fitlm(X, Y(:,1), 'quadratic');
mdl_tensile = fitlm(X, Y(:,2), 'quadratic');
ObjFun = @(x) [predict(mdl_hardness, x), -predict(mdl_tensile, x)];

disp('Initializing NSGA-II Algorithm in Parallel...');

lb = min(X); 
ub = max(X); 

options = optimoptions('gamultiobj', ...
    'PopulationSize', 50, ...
    'MaxGenerations', 100, ...
    'UseParallel', true, ...
    'Display', 'off', ... 
    'OutputFcn', @trackConvergence); 

num_GA_runs = 10;
best_params_all = zeros(num_GA_runs, num_vars);
best_responses_all = zeros(num_GA_runs, 2);
all_Pareto_hardness = [];
all_Pareto_tensile = [];

disp(['Running NSGA-II Iterations ', num2str(num_GA_runs), ' times...']);

for r = 1:num_GA_runs
    fprintf('Executing Run %d of %d...\n', r, num_GA_runs);
    
    [Pareto_Parameters, Pareto_Obj] = gamultiobj(ObjFun, num_vars, [], [], [], [], lb, ub, [], options);
    
    Pareto_hardness = -Pareto_Obj(:,1);
    Pareto_tensile = Pareto_Obj(:,2);

    all_Pareto_hardness = [all_Pareto_hardness; Pareto_hardness];
    all_Pareto_tensile = [all_Pareto_tensile; Pareto_tensile];

    range_H = max(Pareto_hardness) - min(Pareto_hardness);
    range_T = max(Pareto_tensile) - min(Pareto_tensile);
    if range_H == 0, range_H = 1; end 
    if range_T == 0, range_T = 1; end
    
    Norm_H = (Pareto_hardness - min(Pareto_hardness)) ./ range_H;
    Norm_T = (Pareto_tensile - min(Pareto_tensile)) ./ range_T;

    Dist_to_Ideal = sqrt((0 - Norm_H).^2 + (1 - Norm_T).^2);
    [~, best_idx] = min(Dist_to_Ideal);
    
    best_params_all(r, :) = Pareto_Parameters(best_idx, :);
    best_responses_all(r, :) = [Pareto_hardness(best_idx), Pareto_tensile(best_idx)];
end

avg_optimal_params = mean(best_params_all, 1);
avg_optimal_responses = mean(best_responses_all, 1);

disp('----------------------------------------------------');
disp('Best Compromise Solution from Each of the 10 Runs:');
disp('Run |   NT       PS       RD       BT       FR    |      Hardness Tensile');
disp('----------------------------------------------------');
for r = 1:num_GA_runs
    fprintf('%3d | %8.3f %9.3f %9.3f %9.3f %9.3f | %8.3f %9.3f\n', ...
        r, best_params_all(r,:), abs(best_responses_all(r,1)), abs(best_responses_all(r,2)));
end
disp('----------------------------------------------------');
disp('AVERAGE OPTIMAL VALUES ACROSS 10 RUNS:');
fprintf('AVG | %8.3f %9.3f %9.3f %9.3f %9.3f | %8.3f %9.3f\n', ...
    avg_optimal_params, abs(avg_optimal_responses(1)), abs(avg_optimal_responses(2)));
disp('----------------------------------------------------');

figure('Name', 'NSGA-II Pareto Optimal Front (Combined)', 'Color', 'w');
plot(abs(all_Pareto_hardness), abs(all_Pareto_tensile), 'bo', 'MarkerSize', 4, 'MarkerFaceColor', 'b'); hold on;
plot(abs(best_responses_all(:,1)), abs(best_responses_all(:,2)), 'g*', 'MarkerSize', 8, 'LineWidth', 1.5);
plot(abs(avg_optimal_responses(1)), abs(avg_optimal_responses(2)), 'rs', 'MarkerSize', 12, 'MarkerFaceColor', 'r');
xlabel('Hardness (Minimize)'); ylabel('Tensile Strength (Maximize)');
title('Pareto Optimal Fronts across 10 Runs');
legend('All Pareto Solutions', 'Best per Run', 'Average Optimal Value', 'Location', 'Best'); grid on;

try
    conv_F1 = evalin('base', 'history_F1');
    conv_F2 = evalin('base', 'history_F2');
    
    figure('Name', 'NSGA-II Convergence Plot (Last Run)', 'Color', 'w');
    yyaxis left;
    plot(1:length(conv_F1), abs(conv_F1), 'b-', 'LineWidth', 2);
    ylabel('Avg Hardness (Population)');
    
    yyaxis right;
    plot(1:length(conv_F2), abs(conv_F2), 'r-', 'LineWidth', 2);
    ylabel('Avg Tensile Strength (Population)');
    
    xlabel('Generation');
    title('NSGA-II Convergence over Generations (Final Run)');
    grid on;
end

x1_vec = linspace(lb(1), ub(1), 30);
x2_vec = linspace(lb(2), ub(2), 30);
[X1_mat, X2_mat] = meshgrid(x1_vec, x2_vec);
mean_x3 = mean(X(:,3)); mean_x4 = mean(X(:,4)); mean_x5 = mean(X(:,5));
Z_hardness = zeros(size(X1_mat)); Z_tensile = zeros(size(X1_mat));
for i = 1:numel(X1_mat)
    pt = [X1_mat(i), X2_mat(i), mean_x3, mean_x4, mean_x5];
    Z_hardness(i) = predict(mdl_hardness, pt);
    Z_tensile(i) = predict(mdl_tensile, pt);
end

figure('Name', '3D Surface Plots', 'Position', [100 100 1000 400], 'Color', 'w');
subplot(1,2,1);
surf(X1_mat, X2_mat, Z_hardness, 'EdgeColor', 'none'); colormap(gca, parula); colorbar;
xlabel('Nozzle temperature'); ylabel('Print Speed'); zlabel('Hardness');
title('Response Surface: Hardness');
subplot(1,2,2);
surf(X1_mat, X2_mat, Z_tensile, 'EdgeColor', 'none'); colormap(gca, hot); colorbar;
xlabel('Nozzle Temperature'); ylabel('Print Speed'); zlabel('Tensile Strength');
title('Response Surface: Tensile Strength');

function [state, options, optchanged] = trackConvergence(options, state, flag)
    optchanged = false;
    persistent history_F1 history_F2;
    
    switch flag
        case 'init'
            history_F1 = [];
            history_F2 = [];
        case 'iter'
            history_F1 = [history_F1; mean(state.Score(:,1))];
            history_F2 = [history_F2; mean(state.Score(:,2))];
            
            assignin('base', 'history_F1', history_F1);
            assignin('base', 'history_F2', history_F2);
    end
end