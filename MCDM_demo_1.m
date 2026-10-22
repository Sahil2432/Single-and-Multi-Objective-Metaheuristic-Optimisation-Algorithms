disp('=== OPTIMIZATION COMPLETE ===');
disp('Applying TOPSIS to rank the Pareto Front...');

weights = [0.5, 0.5]; 

sq_sum = sqrt(sum(archive_resp.^2, 1));
norm_matrix = archive_resp ./ sq_sum;

weighted_matrix = norm_matrix .* weights;

v_plus = min(weighted_matrix, [], 1);
v_minus = max(weighted_matrix, [], 1);

s_plus = sqrt(sum((weighted_matrix - v_plus).^2, 2));
s_minus = sqrt(sum((weighted_matrix - v_minus).^2, 2));

topsis_score = s_minus ./ (s_plus + s_minus);

final_results_matrix = [archive_pop, archive_resp, topsis_score];
col_names_opt = {'Nozzle_Temp', 'Print_Speed', 'Retraction', 'Bed_Temp', 'Flow_Rate', 'Wear_Rate', 'COF', 'TOPSIS_Score'};

Pareto_Table = array2table(final_results_matrix, 'VariableNames', col_names_opt);
Pareto_Table = sortrows(Pareto_Table, 'TOPSIS_Score', 'descend');

opt_output_file = fullfile(folder_path, 'MOWSO_Pareto_Front_Ranked.xlsx');
writetable(Pareto_Table, opt_output_file);
disp(['Ranked Pareto Front saved to: ', opt_output_file]);

best_compromise_params = table2array(Pareto_Table(1, 1:5));
best_compromise_resp = table2array(Pareto_Table(1, 6:7));
best_score = Pareto_Table.TOPSIS_Score(1);

fprintf('\n--- TOPSIS OPTIMAL COMPROMISE SOLUTION ---\n');
fprintf('Parameters: F1=%.2f, F2=%.2f, F3=%.2f, F4=%.2f, F5=%.2f\n', best_compromise_params);
fprintf('Responses: Wear Rate=%.6f, COF=%.4f \n', best_compromise_resp(1), best_compromise_resp(2));
fprintf('TOPSIS Score: %.4f\n\n', best_score);