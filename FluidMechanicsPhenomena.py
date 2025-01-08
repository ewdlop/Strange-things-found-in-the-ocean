import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const

class FluidMechanicsPhenomena:
    """
    Comprehensive analysis of osmosis, U-tube, and Bernoulli phenomena
    """
    
    def __init__(self, 
                 solute_concentration=0.1,  # mol/L
                 tube_length=1.0,  # meters
                 tube_diameter=0.01  # meters
                 ):
        """
        Initialize parameters for fluid mechanics experiments
        
        Parameters:
        solute_concentration (float): Solute concentration
        tube_length (float): Length of experimental tube
        tube_diameter (float): Diameter of the tube
        """
        # Experimental parameters
        self.solute_concentration = solute_concentration
        self.tube_length = tube_length
        self.tube_diameter = tube_diameter
        
        # Membrane and fluid properties
        self.membrane_properties = {
            'permeability': 1e-10,  # m/s
            'reflection_coefficient': 0.9,
            'thickness': 1e-4  # meters
        }
        
        # Fluid properties (using water as default)
        self.fluid_properties = {
            'density': 1000,  # kg/m³
            'viscosity': 0.001,  # Pa·s
            'molar_mass': 0.018,  # kg/mol
        }
    
    def osmosis_analysis(self):
        """
        Detailed osmosis phenomenon analysis
        
        Returns:
        dict: Osmosis characteristics
        """
        # Van 't Hoff equation for osmotic pressure
        R = const.R  # Universal gas constant
        T = 293.15  # Room temperature (K)
        
        # Osmotic pressure calculation
        osmotic_pressure = self.solute_concentration * R * T
        
        # Membrane flux calculation
        membrane_flux = (
            self.membrane_properties['permeability'] * 
            osmotic_pressure * 
            (1 - self.membrane_properties['reflection_coefficient'])
        )
        
        return {
            'osmotic_pressure': osmotic_pressure,
            'membrane_flux': membrane_flux,
            'solute_concentration': self.solute_concentration
        }
    
    def u_tube_experiment(self, 
                           initial_height_diff=0.1,  # meters
                           fluid_density_diff=50  # kg/m³
                           ):
        """
        U-tube hydrostatic pressure analysis
        
        Parameters:
        initial_height_diff (float): Initial height difference
        fluid_density_diff (float): Density difference between columns
        
        Returns:
        dict: U-tube experiment results
        """
        # Gravitational acceleration
        g = const.g
        
        # Hydrostatic pressure calculation
        pressure_difference = (
            fluid_density_diff * 
            g * 
            initial_height_diff
        )
        
        # Equilibrium height calculation
        def calculate_equilibrium_height(h):
            """
            Calculate height difference at equilibrium
            """
            return (
                fluid_density_diff * h * g - 
                pressure_difference
            )
        
        # Numerical method to find equilibrium
        from scipy.optimize import brentq
        try:
            equilibrium_height = brentq(
                calculate_equilibrium_height, 
                0, 
                initial_height_diff
            )
        except:
            equilibrium_height = initial_height_diff / 2
        
        return {
            'initial_height_difference': initial_height_diff,
            'fluid_density_difference': fluid_density_diff,
            'pressure_difference': pressure_difference,
            'equilibrium_height': equilibrium_height
        }
    
    def bernoulli_principle(self, 
                             flow_velocity=1.0,  # m/s
                             pipe_diameter=0.05  # meters
                             ):
        """
        Bernoulli's principle analysis
        
        Parameters:
        flow_velocity (float): Fluid flow velocity
        pipe_diameter (float): Pipe diameter
        
        Returns:
        dict: Bernoulli principle characteristics
        """
        # Fluid density
        rho = self.fluid_properties['density']
        
        # Pipe cross-sectional area
        pipe_area = np.pi * (pipe_diameter/2)**2
        
        # Volumetric flow rate
        flow_rate = flow_velocity * pipe_area
        
        # Bernoulli equation components
        kinetic_energy_per_volume = 0.5 * rho * flow_velocity**2
        potential_energy_per_volume = rho * const.g * 0  # Assuming no height difference
        
        # Pressure calculation using Bernoulli equation
        # P1 + 1/2ρv1² + ρgh1 = P2 + 1/2ρv2² + ρgh2
        # Simplified for constant height
        pressure_dynamic = 0.5 * rho * flow_velocity**2
        
        return {
            'flow_velocity': flow_velocity,
            'pipe_diameter': pipe_diameter,
            'flow_rate': flow_rate,
            'kinetic_energy_per_volume': kinetic_energy_per_volume,
            'dynamic_pressure': pressure_dynamic
        }
    
    def visualize_results(self):
        """
        Visualize results from different phenomena
        """
        # Compute results
        osmosis_results = self.osmosis_analysis()
        u_tube_results = self.u_tube_experiment()
        bernoulli_results = self.bernoulli_principle()
        
        # Create visualization
        plt.figure(figsize=(15, 5))
        
        # Osmosis Visualization
        plt.subplot(131)
        plt.bar(
            ['Osmotic Pressure', 'Membrane Flux'], 
            [
                osmosis_results['osmotic_pressure'], 
                osmosis_results['membrane_flux']
            ]
        )
        plt.title('Osmosis Characteristics')
        plt.ylabel('Value')
        
        # U-Tube Visualization
        plt.subplot(132)
        plt.bar(
            ['Initial Height', 'Equilibrium Height'], 
            [
                u_tube_results['initial_height_difference'], 
                u_tube_results['equilibrium_height']
            ]
        )
        plt.title('U-Tube Experiment')
        plt.ylabel('Height (m)')
        
        # Bernoulli Principle Visualization
        plt.subplot(133)
        plt.bar(
            ['Flow Velocity', 'Dynamic Pressure'], 
            [
                bernoulli_results['flow_velocity'], 
                bernoulli_results['dynamic_pressure']
            ]
        )
        plt.title("Bernoulli's Principle")
        plt.ylabel('Value')
        
        plt.tight_layout()
        plt.show()

def main():
    # Create fluid mechanics phenomena instance
    fluid_experiment = FluidMechanicsPhenomena(
        solute_concentration=0.1,
        tube_length=1.0,
        tube_diameter=0.01
    )
    
    # Perform analyses
    print("Osmosis Analysis:")
    osmosis_results = fluid_experiment.osmosis_analysis()
    print(f"Osmotic Pressure: {osmosis_results['osmotic_pressure']:.4f} Pa")
    
    print("\nU-Tube Experiment:")
    u_tube_results = fluid_experiment.u_tube_experiment()
    print(f"Equilibrium Height: {u_tube_results['equilibrium_height']:.4f} m")
    
    print("\nBernoulli's Principle:")
    bernoulli_results = fluid_experiment.bernoulli_principle()
    print(f"Flow Rate: {bernoulli_results['flow_rate']:.4f} m³/s")
    
    # Visualize results
    fluid_experiment.visualize_results()

if __name__ == "__main__":
    main()
