import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

class DrainageSoilAbsorptionModel:
    """
    Comprehensive Fluid Dynamics Model for Drainage and Soil Absorption
    
    Key Considerations:
    - Soil porosity
    - Hydraulic conductivity
    - Infiltration rates
    - Soil moisture dynamics
    """
    
    def __init__(self, 
                 soil_type='clay_loam',
                 initial_moisture=0.1,
                 drainage_area=10.0,  # m²
                 initial_water_volume=500.0  # liters
                 ):
        """
        Initialize drainage and soil absorption parameters
        
        Soil Type Properties:
        - Porosity: Void space in soil
        - Hydraulic Conductivity: Water movement through soil
        - Field Capacity: Maximum water retention
        """
        self.soil_properties = {
            'sandy': {
                'porosity': 0.35,
                'hydraulic_conductivity': 0.0144,  # cm/s
                'field_capacity': 0.1,
                'wilting_point': 0.03
            },
            'loam': {
                'porosity': 0.45,
                'hydraulic_conductivity': 0.0036,  # cm/s
                'field_capacity': 0.25,
                'wilting_point': 0.12
            },
            'clay_loam': {
                'porosity': 0.50,
                'hydraulic_conductivity': 0.0012,  # cm/s
                'field_capacity': 0.33,
                'wilting_point': 0.20
            }
        }
        
        self.drainage_area = drainage_area
        self.initial_water_volume = initial_water_volume
        self.current_moisture = initial_moisture
        
        # Select soil properties
        self.soil = self.soil_properties[soil_type]
    
    def infiltration_rate(self, time):
        """
        Calculate infiltration rate using modified Philip's infiltration equation
        
        Parameters:
        time (float): Time since start of drainage (hours)
        
        Returns:
        float: Infiltration rate (mm/hour)
        """
        # Sorptivity coefficient
        S = 0.03 * self.soil['hydraulic_conductivity']
        
        # Time-dependent infiltration rate
        infiltration = S * time**0.5 + self.soil['hydraulic_conductivity'] * time
        
        return max(infiltration, 0)
    
    def water_retention_curve(self, moisture):
        """
        Water retention curve modeling soil water dynamics
        
        Parameters:
        moisture (float): Current soil moisture content
        
        Returns:
        float: Water potential (negative value indicates water retention)
        """
        # Van Genuchten model parameters
        alpha = 0.01  # inverse of air entry pressure
        n = 1.4       # pore size distribution index
        m = 1 - 1/n
        
        # Normalized moisture
        theta_r = self.soil['wilting_point']
        theta_s = self.soil['field_capacity']
        
        # Normalized water content
        S_e = (moisture - theta_r) / (theta_s - theta_r)
        
        # Water potential calculation
        water_potential = -((S_e**(-1/m) - 1)**(1/n)) / alpha
        
        return water_potential
    
    def drainage_simulation(self, simulation_time=24):
        """
        Simulate drainage and soil absorption over time
        
        Parameters:
        simulation_time (float): Total simulation time in hours
        
        Returns:
        dict: Simulation results
        """
        # Time steps
        time_steps = np.linspace(0, simulation_time, 100)
        
        # Storage for results
        water_volume = np.zeros_like(time_steps)
        infiltration_rates = np.zeros_like(time_steps)
        soil_moisture = np.zeros_like(time_steps)
        
        # Initial conditions
        water_volume[0] = self.initial_water_volume
        soil_moisture[0] = self.current_moisture
        
        # Simulation loop
        for i in range(1, len(time_steps)):
            dt = time_steps[i] - time_steps[i-1]
            
            # Infiltration rate
            inf_rate = self.infiltration_rate(time_steps[i])
            infiltration_rates[i] = inf_rate
            
            # Water volume reduction
            water_volume[i] = max(
                water_volume[i-1] - inf_rate * self.drainage_area * dt, 
                0
            )
            
            # Soil moisture update
            soil_moisture[i] = min(
                soil_moisture[i-1] + (inf_rate * dt / self.drainage_area),
                self.soil['field_capacity']
            )
        
        return {
            'time': time_steps,
            'water_volume': water_volume,
            'infiltration_rates': infiltration_rates,
            'soil_moisture': soil_moisture
        }
    
    def visualize_results(self, simulation_results):
        """
        Create visualization of drainage and soil absorption
        
        Parameters:
        simulation_results (dict): Simulation output
        """
        plt.figure(figsize=(15, 5))
        
        # Water Volume Plot
        plt.subplot(131)
        plt.plot(simulation_results['time'], simulation_results['water_volume'])
        plt.title('Water Volume Over Time')
        plt.xlabel('Time (hours)')
        plt.ylabel('Water Volume (L)')
        
        # Infiltration Rate Plot
        plt.subplot(132)
        plt.plot(simulation_results['time'], simulation_results['infiltration_rates'])
        plt.title('Infiltration Rate')
        plt.xlabel('Time (hours)')
        plt.ylabel('Infiltration Rate (mm/hr)')
        
        # Soil Moisture Plot
        plt.subplot(133)
        plt.plot(simulation_results['time'], simulation_results['soil_moisture'])
        plt.title('Soil Moisture Content')
        plt.xlabel('Time (hours)')
        plt.ylabel('Moisture Content')
        
        plt.tight_layout()
        plt.show()

def main():
    # Create drainage model for clay loam soil
    drainage_model = DrainageSoilAbsorptionModel(
        soil_type='clay_loam',
        initial_moisture=0.1,
        drainage_area=10.0,
        initial_water_volume=500.0
    )
    
    # Run drainage simulation
    simulation_results = drainage_model.drainage_simulation(
        simulation_time=24  # 24-hour simulation
    )
    
    # Visualize results
    drainage_model.visualize_results(simulation_results)
    
    # Print key statistics
    print("Drainage Simulation Summary:")
    print(f"Initial Water Volume: {simulation_results['water_volume'][0]:.2f} L")
    print(f"Final Water Volume: {simulation_results['water_volume'][-1]:.2f} L")
    print(f"Peak Infiltration Rate: {max(simulation_results['infiltration_rates']):.4f} mm/hr")
    print(f"Final Soil Moisture: {simulation_results['soil_moisture'][-1]:.4f}")

if __name__ == "__main__":
    main()
