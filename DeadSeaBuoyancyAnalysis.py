import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const

class DeadSeaBuoyancyAnalysis:
    """
    Comprehensive analysis of Dead Sea buoyancy mechanisms
    """
    
    def __init__(self):
        """
        Initialize Dead Sea specific properties
        """
        # Mineral composition of Dead Sea
        self.mineral_composition = {
            'sodium_chloride': 30.4,  # %
            'magnesium_chloride': 23.7,  # %
            'calcium_chloride': 0.4,  # %
            'potassium_chloride': 5.5,  # %
            'other_minerals': 40.0,  # %
        }
        
        # Comparative fluid densities
        self.fluid_densities = {
            'pure_water': 1000,  # kg/m³
            'dead_sea_water': 1240,  # kg/m³
            'ocean_water': 1025,  # kg/m³
        }
    
    def buoyancy_calculation(self, 
                              object_volume=0.001,  # m³
                              object_mass=1.0  # kg
                              ):
        """
        Calculate buoyancy in Dead Sea compared to normal water
        
        Parameters:
        object_volume (float): Volume of the object
        object_mass (float): Mass of the object
        
        Returns:
        dict: Buoyancy characteristics
        """
        # Gravitational acceleration
        g = const.g  # 9.81 m/s²
        
        # Calculate object density
        object_density = object_mass / object_volume
        
        # Buoyant force calculations
        buoyant_force_pure_water = (
            self.fluid_densities['pure_water'] * 
            object_volume * 
            g
        )
        
        buoyant_force_dead_sea = (
            self.fluid_densities['dead_sea_water'] * 
            object_volume * 
            g
        )
        
        # Buoyancy enhancement factor
        buoyancy_enhancement = (
            buoyant_force_dead_sea / buoyant_force_pure_water
        )
        
        return {
            'object_density': object_density,
            'pure_water_density': self.fluid_densities['pure_water'],
            'dead_sea_water_density': self.fluid_densities['dead_sea_water'],
            'buoyant_force_pure_water': buoyant_force_pure_water,
            'buoyant_force_dead_sea': buoyant_force_dead_sea,
            'buoyancy_enhancement_factor': buoyancy_enhancement
        }
    
    def mineral_concentration_analysis(self):
        """
        Analyze mineral concentration effects on buoyancy
        
        Returns:
        dict: Mineral concentration and buoyancy impacts
        """
        # Total dissolved solids calculation
        total_dissolved_solids = sum(self.mineral_composition.values())
        
        # Specific mineral impacts on density
        mineral_density_contributions = {
            'sodium_chloride': 2.16,  # g/cm³
            'magnesium_chloride': 2.32,  # g/cm³
            'calcium_chloride': 2.15,  # g/cm³
            'potassium_chloride': 1.98,  # g/cm³
        }
        
        # Weighted density contribution
        weighted_density_contribution = sum(
            self.mineral_composition[mineral] * 
            mineral_density_contributions.get(mineral, 2.0) 
            for mineral in mineral_density_contributions
        )
        
        return {
            'total_dissolved_solids': total_dissolved_solids,
            'weighted_density_contribution': weighted_density_contribution,
            'mineral_composition': self.mineral_composition
        }
    
    def molecular_interaction_model(self):
        """
        Analyze molecular interactions causing increased buoyancy
        
        Returns:
        dict: Molecular interaction characteristics
        """
        # Simplified molecular interaction model
        interaction_factors = {
            'ionic_strength': 5.0,  # Relative ionic strength
            'hydration_layer_thickness': 0.5,  # nm
            'electrostatic_repulsion': 0.8,  # Relative repulsion factor
        }
        
        # Molecular packing efficiency
        molecular_packing_efficiency = (
            interaction_factors['ionic_strength'] * 
            interaction_factors['electrostatic_repulsion']
        )
        
        return {
            'interaction_factors': interaction_factors,
            'molecular_packing_efficiency': molecular_packing_efficiency
        }
    
    def visualize_results(self):
        """
        Create comprehensive visualization of Dead Sea buoyancy mechanisms
        """
        plt.figure(figsize=(15, 5))
        
        # Density Comparison
        plt.subplot(131)
        densities = [
            self.fluid_densities['pure_water'],
            self.fluid_densities['dead_sea_water'],
            self.fluid_densities['ocean_water']
        ]
        plt.bar(['Pure Water', 'Dead Sea', 'Ocean'], densities)
        plt.title('Water Density Comparison')
        plt.ylabel('Density (kg/m³)')
        
        # Mineral Composition
        plt.subplot(132)
        minerals = list(self.mineral_composition.keys())
        concentrations = list(self.mineral_composition.values())
        plt.pie(concentrations, labels=minerals, autopct='%1.1f%%')
        plt.title('Dead Sea Mineral Composition')
        
        # Buoyancy Enhancement
        plt.subplot(133)
        buoyancy_results = self.buoyancy_calculation()
        plt.bar(
            ['Pure Water Buoyancy', 'Dead Sea Buoyancy'], 
            [
                buoyancy_results['buoyant_force_pure_water'],
                buoyancy_results['buoyant_force_dead_sea']
            ]
        )
        plt.title('Buoyant Force Comparison')
        plt.ylabel('Buoyant Force (N)')
        
        plt.tight_layout()
        plt.show()

def main():
    # Create Dead Sea buoyancy analysis instance
    dead_sea_analysis = DeadSeaBuoyancyAnalysis()
    
    # Buoyancy Calculation
    print("Buoyancy Analysis:")
    buoyancy_results = dead_sea_analysis.buoyancy_calculation()
    print(f"Buoyancy Enhancement Factor: {buoyancy_results['buoyancy_enhancement_factor']:.4f}")
    
    # Mineral Concentration Analysis
    print("\nMineral Concentration:")
    mineral_results = dead_sea_analysis.mineral_concentration_analysis()
    print(f"Total Dissolved Solids: {mineral_results['total_dissolved_solids']:.2f}%")
    
    # Molecular Interaction Model
    print("\nMolecular Interaction:")
    molecular_results = dead_sea_analysis.molecular_interaction_model()
    print(f"Molecular Packing Efficiency: {molecular_results['molecular_packing_efficiency']:.4f}")
    
    # Visualize results
    dead_sea_analysis.visualize_results()

if __name__ == "__main__":
    main()
