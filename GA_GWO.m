% =========================================================================
% 1. Import and Prepare the FDM Dataset
% =========================================================================
% Make sure MATLAB's Current Folder contains this file
filename = 'Tribological_dataset.xlsx'; 
data = readmatrix(filename);

% Extract the 5 Input Parameters (Columns 1 to 5)
% (Nozzle Temp, Print Speed, Retraction Dist, Bed Temp, Flow Rate)
inputs = data(:, 1:5); 

% Extract the 2 Output Parameters (Columns 6 and 7)
% (Frictional force, Coefficient of Friction)
outputs = data(:, 6:7);

% Set target to optimize Frictional force (Column 1 of outputs)
target_output = outputs(:, 1); 

% =========================================================================
% 2. Hybrid Genetic Algorithm - Grey Wolf Optimizer (GA-GWO)
% =========================================================================

% Algorithm Parameters
SearchAgents_no = 30;       % Population size
Max_iter = 100;             % Maximum number of iterations
dim = 5;                    % Number of FDM input parameters
lb = [190, 20, 2, 40, 90];  % Lower bounds
ub = [230, 60, 8, 70, 110]; % Upper bounds
crossover_rate = 0.7;       % GA Crossover Probability
mutation_rate = 0.1;        % GA Mutation Probability

% Initialize Alpha, Beta, and Delta wolves
Alpha_pos = zeros(1, dim);
Alpha_score = inf; 

Beta_pos = zeros(1, dim);
Beta_score = inf;

Delta_pos = zeros(1, dim);
Delta_score = inf;

% Initialize Population randomly within bounds
Positions = initialization(SearchAgents_no, dim, ub, lb);
Convergence_curve = zeros(1, Max_iter);

% Start parallel pool for faster execution
if isempty(gcp('nocreate'))
    parpool('local'); 
end

% Main Optimization Loop
for l = 1:Max_iter
    
    % --- FITNESS EVALUATION (Parallelized) ---
    fitness = zeros(1, SearchAgents_no);
    parfor i = 1:size(Positions, 1)
        % Boundary checking
        Flag4ub = Positions(i,:) > ub;
        Flag4lb = Positions(i,:) < lb;
        Positions(i,:) = (Positions(i,:).*(~(Flag4ub+Flag4lb))) + ub.*Flag4ub + lb.*Flag4lb;               
        
        % Calculate objective function
        fitness(i) = your_objective_function(Positions(i,:), inputs, target_output);
    end
    
    % --- UPDATE ALPHA, BETA, DELTA ---
    for i = 1:SearchAgents_no
        if fitness(i) < Alpha_score 
            Alpha_score = fitness(i); 
            Alpha_pos = Positions(i,:);
        elseif fitness(i) > Alpha_score && fitness(i) < Beta_score 
            Beta_score = fitness(i); 
            Beta_pos = Positions(i,:);
        elseif fitness(i) > Alpha_score && fitness(i) > Beta_score && fitness(i) < Delta_score 
            Delta_score = fitness(i); 
            Delta_pos = Positions(i,:);
        end
    end
    
    % --- GWO POSITION UPDATE ---
    a = 2 - l*((2)/Max_iter); % 'a' decreases linearly from 2 to 0
    
    for i = 1:size(Positions, 1)
        for j = 1:size(Positions, 2)     
            r1 = rand(); r2 = rand();
            A1 = 2*a*r1 - a; C1 = 2*r2;
            D_alpha = abs(C1*Alpha_pos(j) - Positions(i,j));
            X1 = Alpha_pos(j) - A1*D_alpha;
                       
            r1 = rand(); r2 = rand();
            A2 = 2*a*r1 - a; C2 = 2*r2;
            D_beta = abs(C2*Beta_pos(j) - Positions(i,j));
            X2 = Beta_pos(j) - A2*D_beta;       
            
            r1 = rand(); r2 = rand();
            A3 = 2*a*r1 - a; C3 = 2*r2;
            D_delta = abs(C3*Delta_pos(j) - Positions(i,j));
            X3 = Delta_pos(j) - A3*D_delta;             
            
            % New position based on GWO
            Positions(i,j) = (X1 + X2 + X3) / 3;
        end
    end
    
    % --- GA CROSSOVER & MUTATION ---
    for i = 1:size(Positions, 1)
        % Crossover with the Alpha Wolf (The best current solution)
        if rand() < crossover_rate
            crossover_point = randi([1, dim]);
            Positions(i, crossover_point:end) = Alpha_pos(crossover_point:end);
        end
        
        % Mutation
        if rand() < mutation_rate
            mutation_point = randi([1, dim]);
            noise = randn() * (ub(mutation_point) - lb(mutation_point)) * 0.1; 
            Positions(i, mutation_point) = Positions(i, mutation_point) + noise;
        end
    end
    
    Convergence_curve(l) = Alpha_score;
    fprintf('Iteration %d: Best Cost = %f\n', l, Alpha_score);
end

% Plot the convergence curve
figure('Position', [500 500 660 290]);
plot(Convergence_curve, 'LineWidth', 2);
title('GA-GWO Convergence Curve');
xlabel('Iteration');
ylabel('Best Score (RMSE / Error)');
grid on;

% =========================================================================
% Helper Functions
% =========================================================================

function Positions = initialization(SearchAgents_no, dim, ub, lb)
    Boundary_no = size(ub, 2); 
    if Boundary_no == 1
        Positions = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
    else
        Positions = zeros(SearchAgents_no, dim);
        for i = 1:dim
            ub_i = ub(i);
            lb_i = lb(i);
            Positions(:, i) = rand(SearchAgents_no, 1) .* (ub_i - lb_i) + lb_i;
        end
    end
end

function error = your_objective_function(agent_position, inputs, target)
    % IMPORTANT: Replace this logic with your actual ANFIS or prediction model evaluation.
    % The algorithm minimizes this error value.
    
    % Placeholder calculation for demonstration:
    error = sum((agent_position - mean(inputs, 1)).^2); 
end