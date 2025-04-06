import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class InvestmentModel:
    """
    A computational model of two investor populations choosing between speculative stock
    and diversified index investments based on expected utility and mutual expectations.
    """
    
    def __init__(self, 
                 population_size=1000,
                 initial_stock_x_investors_A=0.1,  # Initial fraction of population A investing in stock X
                 initial_stock_x_investors_B=0.1,  # Initial fraction of population B investing in stock X
                 risk_aversion_A=2.0,              # Risk aversion parameter for population A
                 risk_aversion_B=2.0,              # Risk aversion parameter for population B
                 stock_x_return_mean=0.12,         # Mean return of stock X (higher than index)
                 stock_x_return_std=0.30,          # Standard deviation of stock X return (higher risk)
                 index_return_mean=0.08,           # Mean return of diversified index
                 index_return_std=0.15,            # Standard deviation of index return (lower risk)
                 social_influence_factor_A=0.6,    # How much population A is influenced by B's choices
                 social_influence_factor_B=0.6,    # How much population B is influenced by A's choices
                 randomness=0.1,                   # Random noise in decision making
                 max_iterations=100):              # Maximum iterations for simulation
        
        # Population parameters
        self.population_size = population_size
        self.pop_A_stock_x_fraction = initial_stock_x_investors_A
        self.pop_B_stock_x_fraction = initial_stock_x_investors_B
        
        # Utility function parameters (risk aversion)
        self.risk_aversion_A = risk_aversion_A
        self.risk_aversion_B = risk_aversion_B
        
        # Investment return parameters
        self.stock_x_return_mean = stock_x_return_mean
        self.stock_x_return_std = stock_x_return_std
        self.index_return_mean = index_return_mean
        self.index_return_std = index_return_std
        
        # Social influence factors
        self.social_influence_factor_A = social_influence_factor_A
        self.social_influence_factor_B = social_influence_factor_B
        
        # Simulation parameters
        self.randomness = randomness
        self.max_iterations = max_iterations
        
        # History tracking
        self.history_A = [self.pop_A_stock_x_fraction]
        self.history_B = [self.pop_B_stock_x_fraction]
        
    def calculate_expected_utility(self, mean_return, std_return, risk_aversion):
        """
        Calculate expected utility using a constant relative risk aversion (CRRA) utility function.
        U(W) = (W^(1-γ) - 1) / (1-γ), where γ is the risk aversion parameter.
        
        For normally distributed returns, we can approximate expected utility.
        """
        if risk_aversion == 1:  # Log utility case
            expected_utility = np.log(1 + mean_return) - 0.5 * (std_return ** 2) / (1 + mean_return) ** 2
        else:
            # Adjust for risk using a second-order Taylor approximation of the utility function
            expected_utility = (1 + mean_return) ** (1 - risk_aversion) / (1 - risk_aversion) - 1
            expected_utility -= 0.5 * risk_aversion * (std_return ** 2) * (1 + mean_return) ** (1 - 2 * risk_aversion)
        
        return expected_utility
    
    def calculate_social_utility_boost(self, own_fraction, other_fraction, social_factor):
        """
        Calculate utility boost from social influence based on both populations' investment choices.
        Higher fractions of investors in stock X create a "bandwagon effect" or "FOMO" effect.
        """
        # The boost is proportional to both own population and other population investment fractions
        # This creates a reinforcing feedback loop when both populations invest more in stock X
        social_boost = social_factor * (0.5 * own_fraction + 0.5 * other_fraction)
        return social_boost
    
    def update_population_decisions(self):
        """
        Update the fraction of each population investing in stock X based on expected utilities.
        """
        # Calculate base expected utilities for each investment
        stock_x_utility_A = self.calculate_expected_utility(
            self.stock_x_return_mean, self.stock_x_return_std, self.risk_aversion_A)
        
        index_utility_A = self.calculate_expected_utility(
            self.index_return_mean, self.index_return_std, self.risk_aversion_A)
        
        stock_x_utility_B = self.calculate_expected_utility(
            self.stock_x_return_mean, self.stock_x_return_std, self.risk_aversion_B)
        
        index_utility_B = self.calculate_expected_utility(
            self.index_return_mean, self.index_return_std, self.risk_aversion_B)
        
        # Add social influence effects to stock X utilities
        social_boost_A = self.calculate_social_utility_boost(
            self.pop_A_stock_x_fraction, self.pop_B_stock_x_fraction, self.social_influence_factor_A)
        
        social_boost_B = self.calculate_social_utility_boost(
            self.pop_B_stock_x_fraction, self.pop_A_stock_x_fraction, self.social_influence_factor_B)
        
        stock_x_utility_A += social_boost_A
        stock_x_utility_B += social_boost_B
        
        # Calculate probability of choosing stock X using softmax function (logit choice model)
        # This creates a smooth transition rather than abrupt all-or-nothing decisions
        temp = 5.0  # Temperature parameter for softmax (higher = more randomness)
        
        prob_stock_x_A = 1 / (1 + np.exp(-temp * (stock_x_utility_A - index_utility_A)))
        prob_stock_x_B = 1 / (1 + np.exp(-temp * (stock_x_utility_B - index_utility_B)))
        
        # Add some random noise to model heterogeneity and unpredictability
        prob_stock_x_A = np.clip(prob_stock_x_A + np.random.normal(0, self.randomness), 0, 1)
        prob_stock_x_B = np.clip(prob_stock_x_B + np.random.normal(0, self.randomness), 0, 1)
        
        # Update population fractions
        self.pop_A_stock_x_fraction = prob_stock_x_A
        self.pop_B_stock_x_fraction = prob_stock_x_B
        
        # Store history
        self.history_A.append(self.pop_A_stock_x_fraction)
        self.history_B.append(self.pop_B_stock_x_fraction)
    
    def run_simulation(self):
        """
        Run the simulation for the specified number of iterations.
        """
        for _ in range(self.max_iterations):
            self.update_population_decisions()
            
        return self.history_A, self.history_B
    
    def check_bubble_formation(self, threshold=0.7):
        """
        Check if a speculative bubble has formed (both populations heavily invested in stock X).
        """
        if (self.pop_A_stock_x_fraction > threshold and 
            self.pop_B_stock_x_fraction > threshold):
            return True
        return False
    
    def plot_results(self):
        """
        Plot the evolution of investment decisions over time.
        """
        plt.figure(figsize=(12, 8))
        
        # Plot fraction of each population investing in stock X
        plt.subplot(2, 1, 1)
        plt.plot(self.history_A, label='Population A (Stock X Investors)', color='blue')
        plt.plot(self.history_B, label='Population B (Stock X Investors)', color='green')
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% Threshold')
        plt.xlabel('Time Steps')
        plt.ylabel('Fraction Investing in Stock X')
        plt.title('Evolution of Investment Choices Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot the phase space (Population A vs Population B)
        plt.subplot(2, 1, 2)
        plt.plot(self.history_A, self.history_B, 'k-', alpha=0.7)
        plt.plot(self.history_A[0], self.history_B[0], 'go', markersize=10, label='Start')
        plt.plot(self.history_A[-1], self.history_B[-1], 'ro', markersize=10, label='End')
        
        # Add arrows to show direction
        arrow_indices = np.linspace(0, len(self.history_A)-2, min(20, len(self.history_A)-1)).astype(int)
        for i in arrow_indices:
            plt.arrow(self.history_A[i], self.history_B[i], 
                      self.history_A[i+1] - self.history_A[i], 
                      self.history_B[i+1] - self.history_B[i],
                      head_width=0.02, head_length=0.02, fc='k', ec='k', alpha=0.6)
        
        plt.xlabel('Population A (Stock X Investors)')
        plt.ylabel('Population B (Stock X Investors)')
        plt.title('Phase Space: Population A vs Population B')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def create_heatmap(self, param_range, param_name, fixed_params=None):
        """
        Create a heatmap showing how varying a parameter affects bubble formation.
        
        Args:
            param_range: Range of values for the parameter to test
            param_name: Name of the parameter to vary
            fixed_params: Dictionary of other parameters to set
        """
        if fixed_params is None:
            fixed_params = {}
            
        bubble_matrix = np.zeros((len(param_range), len(param_range)))
        
        for i, val_i in enumerate(param_range):
            for j, val_j in enumerate(param_range):
                # Create a new model with the specified parameters
                model_params = {
                    'population_size': self.population_size,
                    'initial_stock_x_investors_A': self.pop_A_stock_x_fraction,
                    'initial_stock_x_investors_B': self.pop_B_stock_x_fraction,
                    'risk_aversion_A': self.risk_aversion_A,
                    'risk_aversion_B': self.risk_aversion_B,
                    'stock_x_return_mean': self.stock_x_return_mean,
                    'stock_x_return_std': self.stock_x_return_std,
                    'index_return_mean': self.index_return_mean,
                    'index_return_std': self.index_return_std,
                    'social_influence_factor_A': self.social_influence_factor_A,
                    'social_influence_factor_B': self.social_influence_factor_B,
                    'randomness': self.randomness,
                    'max_iterations': self.max_iterations
                }
                
                # Update with fixed parameters
                model_params.update(fixed_params)
                
                # Set the varying parameter
                if param_name == 'risk_aversion':
                    model_params['risk_aversion_A'] = val_i
                    model_params['risk_aversion_B'] = val_j
                elif param_name == 'social_influence':
                    model_params['social_influence_factor_A'] = val_i
                    model_params['social_influence_factor_B'] = val_j
                
                # Create and run the model
                model = InvestmentModel(**model_params)
                model.run_simulation()
                
                # Check if a bubble formed
                final_A = model.history_A[-1]
                final_B = model.history_B[-1]
                
                # Measure bubble intensity (0 to 1)
                bubble_intensity = (final_A + final_B) / 2
                bubble_matrix[i, j] = bubble_intensity
        
        # Plot the heatmap using matplotlib instead of seaborn
        plt.figure(figsize=(10, 8))
        
        if param_name == 'risk_aversion':
            x_label = 'Risk Aversion (Population A)'
            y_label = 'Risk Aversion (Population B)'
        elif param_name == 'social_influence':
            x_label = 'Social Influence (Population A)'
            y_label = 'Social Influence (Population B)'
        
        # Create the heatmap using plt.imshow
        plt.imshow(bubble_matrix, cmap='viridis', origin='lower', aspect='auto')
        plt.colorbar(label='Bubble Formation Intensity')
        
        # Set the tick labels
        plt.xticks(range(len(param_range)), [f'{x:.2f}' for x in param_range])
        plt.yticks(range(len(param_range)), [f'{y:.2f}' for y in param_range])
        
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f'Bubble Formation Intensity for Different {param_name.title()} Values')
        plt.tight_layout()
        plt.show()
        
        return bubble_matrix


# Example usage
if __name__ == "__main__":
    # Create model with default parameters
    model = InvestmentModel(
        population_size=1000,
        initial_stock_x_investors_A=0.1,    # Initially only 10% invest in stock X
        initial_stock_x_investors_B=0.1,    # Initially only 10% invest in stock X
        risk_aversion_A=2.0,                # Risk aversion coefficient
        risk_aversion_B=2.0,                # Risk aversion coefficient
        stock_x_return_mean=0.12,           # 12% expected return for stock X
        stock_x_return_std=0.30,            # 30% standard deviation (high risk)
        index_return_mean=0.08,             # 8% expected return for index
        index_return_std=0.15,              # 15% standard deviation (lower risk)
        social_influence_factor_A=0.6,      # Moderate social influence
        social_influence_factor_B=0.6,      # Moderate social influence
        randomness=0.05,                    # Small random noise
        max_iterations=50                   # Run for 50 time steps
    )
    
    # Run simulation
    model.run_simulation()
    
    # Check if a bubble formed
    bubble_formed = model.check_bubble_formation(threshold=0.7)
    print(f"Bubble formed: {bubble_formed}")
    
    # Plot results
    model.plot_results()
    
    # Create heatmap of risk aversion effects
    risk_aversion_range = np.linspace(1.0, 3.0, 10)
    model.create_heatmap(risk_aversion_range, 'risk_aversion')
    
    # Create heatmap of social influence effects
    social_influence_range = np.linspace(0.1, 1.0, 10)
    model.create_heatmap(social_influence_range, 'social_influence')
    
    # Analyze how different parameters affect bubble formation
    # Varying risk aversion while keeping social influence constant
    plt.figure(figsize=(10, 6))
    
    risk_levels = [1.2, 1.5, 2.0, 2.5, 3.0]
    for risk in risk_levels:
        test_model = InvestmentModel(
            risk_aversion_A=risk,
            risk_aversion_B=risk,
            max_iterations=50
        )
        test_model.run_simulation()
        plt.plot(test_model.history_A, label=f'Risk Aversion = {risk}')
    
    plt.title('Effect of Risk Aversion on Stock X Investment (Population A)')
    plt.xlabel('Time Steps')
    plt.ylabel('Fraction Investing in Stock X')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Create an animation of the process
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Run a new simulation for animation
    anim_model = InvestmentModel(max_iterations=100)
    anim_model.run_simulation()
    
    # Set up plot
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Population A (Stock X Investors)')
    ax.set_ylabel('Population B (Stock X Investors)')
    ax.set_title('Dynamic Evolution of Investment Decisions')
    ax.grid(True, alpha=0.3)
    
    line, = ax.plot([], [], 'k-', alpha=0.7)
    point, = ax.plot([], [], 'ro', markersize=8)
    
    # Add contour lines for potential "attractor states"
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    
    # Potential function (simple example - actual dynamics are more complex)
    Z = (X - 0.5)**2 + (Y - 0.5)**2 - 0.3*(X*Y)
    
    contour = ax.contour(X, Y, Z, 15, colors='blue', alpha=0.3)
    
    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point
    
    def animate(i):
        line.set_data(anim_model.history_A[:i+1], anim_model.history_B[:i+1])
        point.set_data(anim_model.history_A[i], anim_model.history_B[i])
        return line, point
    
    anim = FuncAnimation(fig, animate, frames=len(anim_model.history_A),
                          init_func=init, interval=100, blit=True)
    
    plt.show()