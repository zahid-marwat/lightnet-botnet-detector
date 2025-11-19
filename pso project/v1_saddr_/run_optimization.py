import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

class Particle:
    def __init__(self, dimensions, bounds):
        # Initialize particle position and velocity
        self.position = np.zeros(dimensions)
        self.velocity = np.zeros(dimensions)
        self.best_position = np.zeros(dimensions)
        self.best_fitness = -np.inf
        self.fitness = -np.inf
        
        # Initialize random position within bounds
        for i in range(dimensions):
            self.position[i] = np.random.uniform(bounds[i][0], bounds[i][1])
            self.velocity[i] = np.random.uniform(-1, 1)
        
        self.best_position = self.position.copy()

class PSO:
    def __init__(self, dimensions, bounds, num_particles, max_iterations, 
                 X_train, y_train, X_val, y_val):
        self.dimensions = dimensions
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.global_best_position = np.zeros(dimensions)
        self.global_best_fitness = -np.inf
        self.particles = []
        
        # Data for MLP evaluation
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        # Initialize particles
        for i in range(num_particles):
            particle = Particle(dimensions, bounds)
            self.particles.append(particle)
            
        # Initialize convergence history
        self.convergence_history = []
        
    def evaluate_fitness(self, position):
        """Evaluate fitness (AUC) for a particle position"""
        # Map position elements to MLP hyperparameters
        hidden_layer_sizes = tuple([int(position[0])])
        learning_rate_init = position[1]
        alpha = position[2]  # L2 regularization
        
        try:
            # Create and train MLP with the given hyperparameters
            mlp = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                learning_rate_init=learning_rate_init,
                alpha=alpha,
                max_iter=100,  # Limited iterations for faster evaluation
                random_state=42
            )
            
            mlp.fit(self.X_train, self.y_train)
            
            # Predict probabilities for validation set
            y_pred_proba = mlp.predict_proba(self.X_val)
            
            # Calculate AUC (our fitness value)
            if y_pred_proba.shape[1] > 2:  # Multi-class
                # Use one-vs-rest approach for multiclass
                auc = roc_auc_score(self.y_val, y_pred_proba, multi_class='ovr')
            else:  # Binary classification
                auc = roc_auc_score(self.y_val, y_pred_proba[:, 1])
                
            return auc
        except Exception as e:
            print(f"Error during fitness evaluation: {e}")
            return -np.inf
    
    def optimize(self):
        """Run PSO optimization"""
        w = 0.7  # Inertia weight
        c1 = 1.5  # Cognitive weight
        c2 = 1.5  # Social weight
        
        for iteration in range(self.max_iterations):
            for i, particle in enumerate(self.particles):
                # Evaluate fitness
                particle.fitness = self.evaluate_fitness(particle.position)
                
                # Update personal best
                if particle.fitness > particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = particle.position.copy()
                
                # Update global best
                if particle.fitness > self.global_best_fitness:
                    self.global_best_fitness = particle.fitness
                    self.global_best_position = particle.position.copy()
            
            # Update particle velocities and positions
            for particle in self.particles:
                for d in range(self.dimensions):
                    # Update velocity
                    r1, r2 = np.random.random(), np.random.random()
                    
                    cognitive_velocity = c1 * r1 * (particle.best_position[d] - particle.position[d])
                    social_velocity = c2 * r2 * (self.global_best_position[d] - particle.position[d])
                    
                    particle.velocity[d] = w * particle.velocity[d] + cognitive_velocity + social_velocity
                    
                    # Update position
                    particle.position[d] += particle.velocity[d]
                    
                    # Clamp position to bounds
                    if particle.position[d] < self.bounds[d][0]:
                        particle.position[d] = self.bounds[d][0]
                    elif particle.position[d] > self.bounds[d][1]:
                        particle.position[d] = self.bounds[d][1]
            
            # Store best fitness for convergence history
            self.convergence_history.append(self.global_best_fitness)
            print(f"Iteration {iteration+1}/{self.max_iterations}: Best AUC = {self.global_best_fitness:.4f}")
        
        return self.global_best_position, self.global_best_fitness, self.convergence_history

def load_and_preprocess_data(data_path):
    """Load and preprocess IoT cybersecurity data"""
    # Load data - adjust this according to your dataset format
    df = pd.read_csv(data_path)
    
    # Separate features and target
    X = df.drop('label', axis=1)  # Adjust column name as needed
    y = df['label']  
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Further split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42
    ) 
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def run_optimization():
    # Load and preprocess data
    data_path = "path_to_your_dataset.csv"  # Update this path
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data(data_path)
    
    # Define parameter bounds and dimensions
    dimensions = 3
    bounds = [
        (10, 200),     # hidden layer size (neurons in hidden layer)
        (0.0001, 0.1),  # learning rate
        (0.0001, 0.1)  # regularization parameter
    ]
    
    # PSO parameters
    num_particles = 20
    max_iterations = 30
    
    # Initialize and run PSO
    pso = PSO(dimensions, bounds, num_particles, max_iterations, 
              X_train, y_train, X_val, y_val)
    
    best_params, best_fitness, convergence_history = pso.optimize()
    
    # Display optimization results
    print("\nOptimization Results:")
    print(f"Best AUC: {best_fitness:.4f}")
    print(f"Best Parameters:")
    print(f"- Hidden Layer Neurons: {int(best_params[0])}")
    print(f"- Learning Rate: {best_params[1]:.6f}")
    print(f"- Regularization Alpha: {best_params[2]:.6f}")
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_iterations+1), convergence_history, 'b-', linewidth=2)
    plt.title('PSO Convergence Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Best AUC')
    plt.grid(True)
    plt.savefig('pso_convergence.png')
    plt.show()
    
    # Train final model with best parameters
    final_model = MLPClassifier(
        hidden_layer_sizes=(int(best_params[0]),),
        learning_rate_init=best_params[1],
        alpha=best_params[2],
        max_iter=500,
        random_state=42
    )
    
    final_model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred_proba = final_model.predict_proba(X_test)
    
    if y_pred_proba.shape[1] > 2:  # Multi-class
        final_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    else:  # Binary classification
        final_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    
    print(f"\nFinal Test AUC: {final_auc:.4f}")
    
    return final_model, best_params

if __name__ == "__main__":
    final_model, best_params = run_optimization()