import numpy as np
import matplotlib.pyplot as plt

class AircraftCarrierComparison:
    """
    Comprehensive analysis of aircraft carrier displacement
    for USA, China, and India
    """
    
    def __init__(self):
        """
        Define aircraft carrier specifications
        """
        self.carriers = {
            'United States': {
                'class': ['Gerald R. Ford', 'Nimitz'],
                'displacement': {
                    'standard': 100000,  # tons
                    'full_load': 110000,  # tons
                    'details': [
                        {'name': 'USS Gerald R. Ford (CVN-78)', 'displacement': 100000},
                        {'name': 'USS Nimitz', 'displacement': 100000}
                    ]
                },
                'length': 337,  # meters
                'width': 78,  # meters
                'aircraft_capacity': 75
            },
            'China': {
                'class': ['Type 003'],
                'displacement': {
                    'standard': 80000,  # tons
                    'full_load': 85000,  # tons
                    'details': [
                        {'name': 'Fujian (CV-18)', 'displacement': 80000}
                    ]
                },
                'length': 316,  # meters
                'width': 74,  # meters
                'aircraft_capacity': 60
            },
            'India': {
                'class': ['Vikrant'],
                'displacement': {
                    'standard': 37500,  # tons
                    'full_load': 40000,  # tons
                    'details': [
                        {'name': 'INS Vikrant', 'displacement': 37500}
                    ]
                },
                'length': 262,  # meters
                'width': 62,  # meters
                'aircraft_capacity': 30
            }
        }
    
    def displacement_comparison(self):
        """
        Compare aircraft carrier displacements
        
        Returns:
        dict: Displacement comparison details
        """
        # Extract displacement data
        countries = list(self.carriers.keys())
        standard_displacement = [
            self.carriers[country]['displacement']['standard'] 
            for country in countries
        ]
        full_load_displacement = [
            self.carriers[country]['displacement']['full_load'] 
            for country in countries
        ]
        
        return {
            'countries': countries,
            'standard_displacement': standard_displacement,
            'full_load_displacement': full_load_displacement
        }
    
    def naval_capability_analysis(self):
        """
        Analyze naval capabilities beyond displacement
        
        Returns:
        dict: Comprehensive naval capability metrics
        """
        # Compute naval capability index
        capability_index = {}
        for country, data in self.carriers.items():
            # Simple capability index calculation
            capability_index[country] = (
                data['displacement']['full_load'] *
                data['aircraft_capacity'] / 
                (data['length'] * data['width'])
            )
        
        return {
            'capability_index': capability_index,
            'aircraft_capacities': {
                country: data['aircraft_capacity'] 
                for country, data in self.carriers.items()
            }
        }
    
    def visualize_comparison(self):
        """
        Create comprehensive visualization of carrier comparisons
        """
        # Displacement comparison
        disp_data = self.displacement_comparison()
        
        # Create multi-panel visualization
        plt.figure(figsize=(15, 5))
        
        # Displacement Comparison
        plt.subplot(131)
        plt.bar(disp_data['countries'], disp_data['standard_displacement'])
        plt.title('Standard Displacement')
        plt.ylabel('Displacement (tons)')
        plt.xticks(rotation=45)
        
        # Full Load Displacement
        plt.subplot(132)
        plt.bar(disp_data['countries'], disp_data['full_load_displacement'])
        plt.title('Full Load Displacement')
        plt.ylabel('Displacement (tons)')
        plt.xticks(rotation=45)
        
        # Naval Capability Analysis
        capability_data = self.naval_capability_analysis()
        plt.subplot(133)
        plt.bar(
            list(capability_data['capability_index'].keys()),
            list(capability_data['capability_index'].values())
        )
        plt.title('Naval Capability Index')
        plt.ylabel('Capability Score')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def detailed_report(self):
        """
        Generate detailed textual report of aircraft carrier specifications
        """
        report = "Aircraft Carrier Comparative Analysis\n"
        report += "=" * 40 + "\n\n"
        
        for country, data in self.carriers.items():
            report += f"{country} Aircraft Carriers:\n"
            report += f"Classes: {', '.join(data['class'])}\n"
            report += f"Standard Displacement: {data['displacement']['standard']} tons\n"
            report += f"Full Load Displacement: {data['displacement']['full_load']} tons\n"
            report += f"Length: {data['length']} meters\n"
            report += f"Width: {data['width']} meters\n"
            report += f"Aircraft Capacity: {data['aircraft_capacity']} aircraft\n\n"
        
        return report

def main():
    # Create carrier comparison instance
    carrier_analysis = AircraftCarrierComparison()
    
    # Print detailed report
    print(carrier_analysis.detailed_report())
    
    # Compute displacement comparison
    disp_comp = carrier_analysis.displacement_comparison()
    print("Displacement Comparison:")
    for country, disp in zip(disp_comp['countries'], disp_comp['standard_displacement']):
        print(f"{country}: {disp} tons")
    
    # Naval capability analysis
    capability = carrier_analysis.naval_capability_analysis()
    print("\nNaval Capability Index:")
    for country, index in capability['capability_index'].items():
        print(f"{country}: {index:.2f}")
    
    # Visualize comparison
    carrier_analysis.visualize_comparison()

if __name__ == "__main__":
    main()
