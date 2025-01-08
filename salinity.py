import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const

class FluidPropertiesAnalysis:
    """
    Comprehensive analysis of buoyancy, water density, 
    and colligative properties
    """
    
    def __init__(self, 
                 water_temperature=20,  # °C
                 salinity=0.0  # salt concentration (%)
                 ):
        """
        Initialize fluid properties
        
        Parameters:
        water_temperature (float): Water temperature in Celsius
        salinity (float): Salt concentration percentage
        """
        self.temperature = water_temperature
        self.salinity = salinity
        
        # Fundamental water properties
        self.water_properties = {
            'pure_water_density': self.calculate_water_density(),
            'surface_tension': 0.072,  # N/m at 20°C
            'specific_heat_capacity': 4186  # J/(kg·K)
        }
    
    def calculate_water_density(self):
        """
        Calculate water density considering temperature and salinity
        
        Returns:
        float: Water density in kg/m³
        """
        # Base density of pure water at 4°C
        base_density = 1000  # kg/m³
        
        # Temperature correction
        # Approximate linear model
        temp_correction = -0.21 * (self.temperature - 4)
        
        # Salinity correction
        # Approximate linear model
        salinity_correction = 0.7 * self.salinity
        
        return base_density + temp_correction - salinity_correction
    
    def buoyancy_analysis(self, 
                           object_volume=0.001,  # m³
                           object_mass=1.0  # kg
                           ):
        """
        Analyze buoyancy of an object in water
        
        Parameters:
        object_volume (float): Volume of the object in m³
        object_mass (float): Mass of the object in kg
        
        Returns:
        dict: Buoyancy characteristics
        """
        # Gravitational acceleration
        g = const.g  # 9.81 m/s²
        
        # Calculate object density
        object_density = object_mass / object_volume
        
        # Buoyant force calculation
        buoyant_force = (
            self.water_properties['pure_water_density'] * 
            object_volume * 
            g
        )
        
        # Determine buoyancy state
        if object_density < self.water_properties['pure_water_density']:
            buoyancy_state = "Floats"
        elif object_density > self.water_properties['pure_water_density']:
            buoyancy_state = "Sinks"
        else:
            buoyancy_state = "Neutrally Buoyant"
        
        return {
            'object_density': object_density,
            'water_density': self.water_properties['pure_water_density'],
            'buoyant_force': buoyant_force,
            'buoyancy_state': buoyancy_state
        }
    
    def colligative_properties_analysis(self, 
                                        solute_type='NaCl',
                                        solute_concentration=0.1  # mol/L
                                        ):
        """
        Analyze colligative properties and melting point depression
        
        Parameters:
        solute_type (str): Type of solute
        solute_concentration (float): Solute concentration in mol/L
        
        Returns:
        dict: Colligative properties analysis
        """
        # Solute-specific constants
        solute_properties = {
            'NaCl': {
                'dissociation_factor': 2,  # Na+ and Cl- ions
                'molar_mass': 58.44,  # g/mol
            },
            'Sucrose': {
                'dissociation_factor': 1,  # Non-electrolyte
                'molar_mass': 342.30,  # g/mol
            }
        }
        
        # Select solute properties
        solute = solute_properties.get(solute_type, solute_properties['NaCl'])
        
        # Molality calculation (mol solute per kg solvent)
        # Assuming water as solvent with density 1 kg/L
        molality = solute_concentration
        
        # Freezing point depression constant for water
        K_f = 1.86  # °C·kg/mol
        
        # Melting point depression calculation
        melting_point_depression = (
            solute['dissociation_factor'] * 
            K_f * 
            molality
        )
        
        return {
            'solute_type': solute_type,
            'concentration': solute_concentration,
            'melting_point_depression': melting_point_depression,
            'dissociation_factor': solute['dissociation_factor']
        }
    
    def visualize_results(self):
        """
        Visualize key fluid properties and colligative effects
        """
        plt.figure(figsize=(15, 5))
        
        # Buoyancy Analysis
        plt.subplot(131)
        buoyancy_results = self.buoyancy_analysis()
        plt.bar(
            ['Object Density', 'Water Density'], 
            [
                buoyancy_results['object_density'], 
                buoyancy_results['water_density']
            ]
        )
        plt.title('Buoyancy Comparison')
        plt.ylabel('Density (kg/m³)')
        
        # Water Density vs Temperature
        plt.subplot(132)
        temperatures = np.linspace(0, 30, 50)
        densities = [
            self.calculate_water_density() 
            for self.temperature in temperatures
        ]
        plt.plot(temperatures, densities)
        plt.title('Water Density vs Temperature')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Density (kg/m³)')
        
        # Melting Point Depression
        plt.subplot(133)
        mp_results = self.colligative_properties_analysis()
        plt.bar(
            ['NaCl Concentration', 'Melting Point Depression'], 
            [
                mp_results['concentration'], 
                mp_results['melting_point_depression']
            ]
        )
        plt.title('Melting Point Depression')
        plt.ylabel('Concentration / Depression (°C)')
        
        plt.tight_layout()
        plt.show()

def main():
    # Create fluid properties analysis instance
    fluid_analysis = FluidPropertiesAnalysis(
        water_temperature=20,
        salinity=0.1
    )
    
    # Buoyancy Analysis
    print("Buoyancy Analysis:")
    buoyancy_results = fluid_analysis.buoyancy_analysis()
    print(f"Buoyancy State: {buoyancy_results['buoyancy_state']}")
    print(f"Buoyant Force: {buoyancy_results['buoyant_force']:.4f} N")
    
    # Colligative Properties Analysis
    print("\nColligative Properties Analysis:")
    colligative_results = fluid_analysis.colligative_properties_analysis()
    print(f"Melting Point Depression: {colligative_results['melting_point_depression']:.4f} °C")
    
    # Visualize results
    fluid_analysis.visualize_results()

if __name__ == "__main__":
    main()
