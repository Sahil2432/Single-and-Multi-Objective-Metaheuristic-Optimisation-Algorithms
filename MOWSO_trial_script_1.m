
folder_path = "C:\Users\sahil\OneDrive\Desktop\DOE_thesis\New_DOE"; 
file_name = 'New_DOE_Optimisation_dataset.xlsx';         
full_file_path = fullfile(folder_path, file_name);

disp('Loading 45-run Excel dataset...');
dataset = readmatrix(full_file_path); 

X_train = dataset(:, 1:5); 
Y_wear  = dataset(:, 6);   
Y_cof   = dataset(:, 7);   

disp('Training Quadratic Response Surface Models...');
mdl_wear = fitlm(X_train, Y_wear, 'quadratic');
mdl_cof  = fitlm(X_train, Y_cof, 'quadratic');

disp('Generating predictions for the original 45 DoE runs...');
Y_pred_wear = predict(mdl_wear, X_train);
Y_pred_cof  = predict(mdl_cof, X_train);

DoE_Validation_Matrix = [X_train, Y_wear, Y_pred_wear, Y_cof, Y_pred_cof];
col_names_val = {'F1', 'F2', 'F3', 'F4', 'F5', 'Actual_Wear', 'Pred_Wear', 'Actual_COF', 'Pred_COF'};
Validation_Table = array2table(DoE_Validation_Matrix, 'VariableNames', col_names_val);
val_output_file = fullfile(folder_path, 'MOWSO_45_Run_Predictions.xlsx');
writetable(Validation_Table, val_output_file);
disp(['45-Run Predictions saved to: ', val_output_file]);

disp('Starting Vectorized Continuous MOWSO Optimization...');
dim = 5; 

lb = min(X_train); 
ub = max(X_train); 

num_agents = 100; 
max_iter = 150;   

population = repmat(lb, num_agents, 1) + rand(num_agents, dim) .* repmat((ub - lb), num_agents, 1);
velocities = zeros(num_agents, dim);

responses = evaluate_using_model(population, mdl_wear, mdl_cof);
[archive_pop, archive_resp] = get_non_dominated_archive(population, responses);

for iter = 1:max_iter
    p1 = 1.5 - (1.5 - 0.5) * exp(-(4 * iter / max_iter)^2);
    
    for i = 1:num_agents
        global_best_idx = randi(size(archive_pop, 1));
        global_best_pos = archive_pop(global_best_idx, :);
        
        c1 = rand();
        mu = 0.5;
        
       
        velocities(i, :) = mu * (velocities(i, :) + p1 * (global_best_pos - population(i, :)) * c1);
        population(i, :) = population(i, :) + velocities(i, :);
        
       
        population(i, :) = max(population(i, :), lb);
        population(i, :) = min(population(i, :), ub);
    end
       
    responses = evaluate_using_model(population, mdl_wear, mdl_cof);
    
      combined_pop = [population; archive_pop];
    combined_resp = [responses; archive_resp];
    [archive_pop, archive_resp] = get_non_dominated_archive(combined_pop, combined_resp);

    if size(archive_pop, 1) > 100
        keep_idx = randperm(size(archive_pop, 1), 100);
        archive_pop = archive_pop(keep_idx, :);
        archive_resp = archive_resp(keep_idx, :);
    end
end
disp('=== OPTIMIZATION COMPLETE ===');

final_results_matrix = [archive_pop, archive_resp];
col_names_opt = {'Nozzle_Temp', 'Print_Speed', 'Retraction', 'Bed_Temp', 'Flow_Rate', 'Wear_Rate', 'COF'};
Pareto_Table = array2table(final_results_matrix, 'VariableNames', col_names_opt);
opt_output_file = fullfile(folder_path, 'MOWSO_Pareto_Front.xlsx');
writetable(Pareto_Table, opt_output_file);

[~, best_wear_idx] = min(archive_resp(:, 1));
[~, best_cof_idx] = min(archive_resp(:, 2));

fprintf('\n--- ABSOLUTE OPTIMAL PARAMETERS ---\n');
fprintf('BEST WEAR RATE (%.6f): F1=%.2f, F2=%.2f, F3=%.2f, F4=%.2f, F5=%.2f (COF=%.4f)\n', ...
    archive_resp(best_wear_idx, 1), archive_pop(best_wear_idx, :), archive_resp(best_wear_idx, 2));
fprintf('BEST COF (%.4f): F1=%.2f, F2=%.2f, F3=%.2f, F4=%.2f, F5=%.2f (Wear=%.6f)\n\n', ...
    archive_resp(best_cof_idx, 2), archive_pop(best_cof_idx, :), archive_resp(best_cof_idx, 1));

figure('Name', 'RSM Validation: Predicted vs Actual', 'Color', 'w', 'Position', [50, 100, 900, 400]);
subplot(1,2,1);
scatter(Y_wear, Y_pred_wear, 50, 'b', 'filled'); hold on;
plot([min(Y_wear) max(Y_wear)], [min(Y_wear) max(Y_wear)], 'r--', 'LineWidth', 1.5);
title('Validation: Wear Rate'); xlabel('Actual Wear'); ylabel('Predicted Wear'); grid on;
subplot(1,2,2);
scatter(Y_cof, Y_pred_cof, 50, 'g', 'filled'); hold on;
plot([min(Y_cof) max(Y_cof)], [min(Y_cof) max(Y_cof)], 'r--', 'LineWidth', 1.5);
title('Validation: COF'); xlabel('Actual COF'); ylabel('Predicted COF'); grid on;

figure('Name', 'MOWSO Pareto Front', 'Color', 'w', 'Position', [100, 150, 600, 500]);
scatter(archive_resp(:, 1), archive_resp(:, 2), 70, 'b', 'filled', 'MarkerEdgeColor', 'k');
title('MOWSO Pareto Front: Wear vs. COF');
xlabel('Specific Wear Rate (Minimize)');
ylabel('Coefficient of Friction (Minimize)');
grid on; set(gca, 'FontSize', 12, 'LineWidth', 1.2);

disp('Generating Vectorized 3D surface plots...');
factor_names = {'Nozzle Temperature', 'Print Speed', 'Retraction Distance', 'Bed Temperature', 'Flow Rate'};
combinations = nchoosek(1:5, 2);
medians = median(X_train); 

for c = 1:size(combinations, 1)
    f_x = combinations(c, 1);
    f_y = combinations(c, 2);
    
    [X_grid, Y_grid] = meshgrid(linspace(lb(f_x), ub(f_x), 30), ...
                                linspace(lb(f_y), ub(f_y), 30));
                            
        num_elements = numel(X_grid);
    input_matrix = repmat(medians, num_elements, 1);
    input_matrix(:, f_x) = X_grid(:);
    input_matrix(:, f_y) = Y_grid(:);
    
    WR_flat = predict(mdl_wear, input_matrix);
    COF_flat = predict(mdl_cof, input_matrix);
    
    WR_surface = reshape(WR_flat, size(X_grid));
    COF_surface = reshape(COF_flat, size(X_grid));
    
    fig_name = sprintf('3D Surface: %s vs %s', factor_names{f_x}, factor_names{f_y});
    figure('Name', fig_name, 'Color', 'w', 'Position', [150 + c*20, 150 + c*20, 1000, 400]);
    
    subplot(1, 2, 1);
    surf(X_grid, Y_grid, WR_surface, 'EdgeColor', 'none', 'FaceAlpha', 0.9);
    title(sprintf('Wear Rate\n(%s vs %s)', factor_names{f_x}, factor_names{f_y}));
    xlabel(factor_names{f_x}); ylabel(factor_names{f_y}); zlabel('Wear');
    colormap(jet); colorbar; view(-45, 25);
    
    subplot(1, 2, 2);
    surf(X_grid, Y_grid, COF_surface, 'EdgeColor', 'none', 'FaceAlpha', 0.9);
    title(sprintf('COF\n(%s vs %s)', factor_names{f_x}, factor_names{f_y}));
    xlabel(factor_names{f_x}); ylabel(factor_names{f_y}); zlabel('COF');
    colormap(jet); colorbar; view(-45, 25);
    
    drawnow; 
end
disp('All 3D Surface Plots generated.');

function res = evaluate_using_model(pop_matrix, mdl_wear, mdl_cof)
   
    pred_wear = predict(mdl_wear, pop_matrix);
    pred_cof = predict(mdl_cof, pop_matrix);
    pred_wear = max(pred_wear, 0); 
    res = [pred_wear, pred_cof];
end

function [arc_pop, arc_resp] = get_non_dominated_archive(pop, resp)
    N = size(pop, 1);
    is_dominated = false(N, 1);
    for i = 1:N
        for j = 1:N
            if i ~= j
                if all(resp(j, :) <= resp(i, :)) && any(resp(j, :) < resp(i, :))
                    is_dominated(i) = true;
                    break;
                end
            end
        end
    end
    arc_pop = pop(~is_dominated, :);
    arc_resp = resp(~is_dominated, :);
    [~,unique_idx, ~] = unique(arc_resp, 'rows');
    arc_pop = arc_pop(unique_idx, :);
    arc_resp = arc_resp(unique_idx, :);
end