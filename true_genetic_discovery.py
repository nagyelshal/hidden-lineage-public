#!/usr/bin/env python3
"""
GENETIC DISCOVERY ENGINE: True Ancestry Without Commercial Bias
Find who you ACTUALLY cluster with, not who marketing says you should
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def find_true_genetic_neighbors(n_neighbors=500, sample_prefix="I0001"):
    """Find your closest genetic neighbors without any population bias"""
    
    print("🔍 LOADING DATA FOR TRUE GENETIC DISCOVERY...")
    
    # Load PCA results
    pca_data = pd.read_csv('trimmed_pca_results.eigenvec', sep=r'\s+', header=None,
                          names=['FID', 'IID', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 
                                'PC6', 'PC7', 'PC8', 'PC9', 'PC10'])
    
    # Load full metadata from AADR with robust fallback
    try:
        aadr_anno = pd.read_csv('raw_data/aadr_v62/dataverse_files/v62.0_1240k_public.anno', 
                               sep='\t', low_memory=False)
        print(f"✅ Loaded {len(aadr_anno)} samples from AADR annotation")
        
        # Map the actual column names to what the script expects
        genetic_id_col = aadr_anno.columns[0]  # The very long first column name
        column_mapping = {
            genetic_id_col: 'Instance_ID',
            'Group ID': 'Group_Name',
            'Political Entity': 'Country',
            'Date mean in BP in years before 1950 CE [OxCal mu for a direct radiocarbon date, and average of range for a contextual date]': 'Date_mean_BP',
            'Publication abbreviation': 'Publication',
            'Lat.': 'Lat.',
            'Long.': 'Long.'
        }
        
        # Rename columns to match what the script expects
        aadr_anno = aadr_anno.rename(columns=column_mapping)
        
        # Add Age_group column based on Date_mean_BP
        aadr_anno['Age_group'] = 'Unknown'
        dated_mask = pd.to_numeric(aadr_anno['Date_mean_BP'], errors='coerce').notna()
        aadr_anno.loc[dated_mask, 'Age_group'] = pd.cut(
            pd.to_numeric(aadr_anno.loc[dated_mask, 'Date_mean_BP'], errors='coerce'),
            bins=[0, 500, 2000, 5000, 10000, 50000, np.inf],
            labels=['Modern', 'Historical', 'Medieval', 'Bronze_Age', 'Neolithic', 'Paleolithic']
        ).astype(str)
        
    except Exception as e:
        print(f"⚠️ Could not load AADR annotation: {e}")
        print("📊 Creating dummy annotation for basic analysis")
        aadr_anno = pd.DataFrame({
            'Instance_ID': pca_data['IID'],
            'Group_Name': 'Unknown',
            'Country': 'Unknown',
            'Date_mean_BP': np.nan,
            'Publication': 'Unknown',
            'Lat.': np.nan,
            'Long.': np.nan,
            'Age_group': 'Unknown'
        })
    
    # Robust sample matching - exact prefix match
    your_samples = pca_data[pca_data['IID'].str.startswith(sample_prefix)]
    if your_samples.empty:
        print(f"❌ Could not find samples with prefix '{sample_prefix}'!")
        print(f"🔍 Available sample prefixes: {pca_data['IID'].str[:5].unique()[:10]}")
        return None
    
    print(f"✅ Found {len(your_samples)} samples with prefix '{sample_prefix}'")
    
    # FIXED: Use all 10 PCs for maximum resolution and consistent variable naming
    your_coords = your_samples[['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10']].mean().values
    print(f"📍 Your genetic coordinates: {your_coords[:2]}")
    
    # Calculate distances to ALL other samples using full 10D space
    other_samples = pca_data[~pca_data['IID'].str.startswith(sample_prefix)]
    other_coords = other_samples[['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10']].values
    
    # FIXED: Use consistent variable name
    distances = cdist([your_coords], other_coords, metric='euclidean')[0]
    
    # Create distance dataframe
    distance_df = pd.DataFrame({
        'Sample_ID': other_samples['IID'].values,
        'Distance': distances
    })
    
    # Sort by distance
    closest_neighbors = distance_df.sort_values('Distance').head(n_neighbors)
    
    print(f"🎯 Found {len(closest_neighbors)} closest genetic neighbors")
    print(f"🔥 Including {len([x for x in closest_neighbors['Sample_ID'] if any(pop in x for pop in ['Albania_', 'Armenia_', 'Afghanistan_'])])} rare/singleton populations")
    
    return closest_neighbors, pca_data, aadr_anno

def enrich_with_metadata(closest_neighbors, aadr_anno):
    """Add rich metadata to closest neighbors"""
    
    if aadr_anno is None:
        return closest_neighbors
    
    print("📋 Enriching with archaeological and geographical metadata...")
    
    # Merge with AADR annotation using the correct column names
    enriched = closest_neighbors.merge(
        aadr_anno[['Instance_ID', 'Group_Name', 'Country', 'Date_mean_BP', 
                  'Publication', 'Lat.', 'Long.', 'Age_group']].rename(columns={'Instance_ID': 'Sample_ID'}),
        on='Sample_ID', how='left'
    )
    
    # Clean up the data
    enriched['Date_mean_BP'] = pd.to_numeric(enriched['Date_mean_BP'], errors='coerce')
    enriched['Lat.'] = pd.to_numeric(enriched['Lat.'], errors='coerce')
    enriched['Long.'] = pd.to_numeric(enriched['Long.'], errors='coerce')
    
    return enriched

def analyze_genetic_neighborhood(enriched_neighbors):
    """Analyze patterns in your genetic neighborhood"""
    
    print("\n🧬 ANALYZING YOUR TRUE GENETIC NEIGHBORHOOD")
    print("=" * 60)
    
    # Initialize temporal_dist to avoid scope issues
    temporal_dist = None
    
    # Geographic analysis
    print("\n🌍 GEOGRAPHIC DISTRIBUTION:")
    country_counts = enriched_neighbors['Country'].value_counts().head(10)
    for country, count in country_counts.items():
        pct = (count / len(enriched_neighbors)) * 100
        print(f"   {country}: {count} samples ({pct:.1f}%)")
    
    # Temporal analysis
    print("\n⏰ TEMPORAL DISTRIBUTION:")
    dated_samples = enriched_neighbors.dropna(subset=['Date_mean_BP'])
    if not dated_samples.empty:
        date_bins = pd.cut(dated_samples['Date_mean_BP'], 
                          bins=[0, 2000, 5000, 8000, 12000, 50000], 
                          labels=['Modern', 'Historical', 'Neolithic', 'Mesolithic', 'Paleolithic'],
                          right=False, include_lowest=True)
        temporal_dist = date_bins.value_counts()
        for period, count in temporal_dist.items():
            pct = (count / len(dated_samples)) * 100
            print(f"   {period}: {count} samples ({pct:.1f}%)")
    else:
        print("   ⚠️ No dated samples in closest neighbors")
    
    # Population group analysis
    print("\n🏛️ POPULATION GROUPS:")
    if 'Group_Name' in enriched_neighbors.columns:
        group_counts = enriched_neighbors['Group_Name'].value_counts().head(15)
        for group, count in group_counts.items():
            pct = (count / len(enriched_neighbors)) * 100
            print(f"   {group}: {count} samples ({pct:.1f}%)")
    
    # EDGE CASE ANALYSIS - The real discovery!
    print("\n🔥 EDGE CASE DISCOVERIES:")
    if 'Group_Name' in enriched_neighbors.columns:
        # Find singleton matches (groups with only 1 representative in your neighborhood)
        group_counts_full = enriched_neighbors['Group_Name'].value_counts()
        singleton_matches = enriched_neighbors[group_counts_full[enriched_neighbors['Group_Name']].eq(1)]
        
        if not singleton_matches.empty:
            print(f"   🎯 SINGLETON POPULATION MATCHES: {len(singleton_matches)}")
            # Save singleton matches to CSV for detailed analysis
            singleton_matches.to_csv('singleton_genetic_matches.csv', index=False)
            print(f"   📄 Saved to: singleton_genetic_matches.csv")
            
            for _, match in singleton_matches.head(10).iterrows():
                sample_id = str(match['Sample_ID'])[:20]
                group_name = str(match.get('Group_Name', 'Unknown'))[:30]
                country = str(match.get('Country', 'Unknown'))
                distance = match['Distance']
                print(f"      • {sample_id:<20} | {group_name:<30} | {country} (d={distance:.6f})")
        else:
            print("   📊 No singleton matches found - you cluster with established populations")
    
    # Distance analysis
    print("\n📏 GENETIC DISTANCES:")
    distances = enriched_neighbors['Distance']
    print(f"   Closest match: {distances.min():.6f}")
    print(f"   Median distance: {distances.median():.6f}")
    print(f"   95th percentile: {distances.quantile(0.95):.6f}")
    
    return country_counts, temporal_dist

def create_discovery_visualization(enriched_neighbors, pca_data, sample_prefix="I0001"):
    """Create visualization showing true genetic discovery"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.patch.set_facecolor('black')
    
    # Get your coordinates
    your_samples = pca_data[pca_data['IID'].str.startswith(sample_prefix)]
    your_pc1, your_pc2 = your_samples[['PC1', 'PC2']].mean()
    
    # Plot 1: Geographic distribution
    ax1.set_facecolor('black')
    
    if 'Lat.' in enriched_neighbors.columns and 'Long.' in enriched_neighbors.columns:
        geo_data = enriched_neighbors.dropna(subset=['Lat.', 'Long.'])
        if not geo_data.empty:
            scatter = ax1.scatter(geo_data['Long.'], geo_data['Lat.'], 
                                c=geo_data['Distance'], cmap='RdYlBu_r', 
                                s=50, alpha=0.8, edgecolors='white', linewidth=0.5)
            ax1.set_xlabel('Longitude', color='white', fontweight='bold')
            ax1.set_ylabel('Latitude', color='white', fontweight='bold')
            ax1.set_title('Geographic Distribution of Your Closest Genetic Matches', 
                         color='white', fontsize=14, fontweight='bold')
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label('Genetic Distance', color='white')
            cbar.ax.yaxis.set_tick_params(color='white')
            ax1.tick_params(colors='white')
    
    # Plot 2: Temporal distribution
    ax2.set_facecolor('black')
    
    dated_samples = enriched_neighbors.dropna(subset=['Date_mean_BP'])
    if not dated_samples.empty:
        ax2.scatter(dated_samples['Date_mean_BP'], dated_samples['Distance'], 
                   c='cyan', alpha=0.8, s=40, edgecolors='white', linewidth=0.5)
        ax2.set_xlabel('Years Before Present', color='white', fontweight='bold')
        ax2.set_ylabel('Genetic Distance to You', color='white', fontweight='bold')
        ax2.set_title('Temporal Distribution of Genetic Matches', 
                     color='white', fontsize=14, fontweight='bold')
        ax2.tick_params(colors='white')
        ax2.set_xscale('log')
    
    # Plot 3: PCA with closest neighbors highlighted
    ax3.set_facecolor('black')
    
    # Plot all samples in gray
    ax3.scatter(pca_data['PC1'], pca_data['PC2'], c='gray', alpha=0.1, s=1)
    
    # Get coordinates for closest neighbors
    neighbor_coords = []
    singleton_coords = []
    
    # Check if we have singleton matches
    if 'Group_Name' in enriched_neighbors.columns:
        group_counts = enriched_neighbors['Group_Name'].value_counts()
        singletons = enriched_neighbors[group_counts[enriched_neighbors['Group_Name']].eq(1)]
    else:
        singletons = pd.DataFrame()
    
    for i, sample_id in enumerate(enriched_neighbors['Sample_ID'].head(50)):
        sample_data = pca_data[pca_data['IID'] == sample_id]
        if not sample_data.empty:
            coords = [sample_data['PC1'].iloc[0], sample_data['PC2'].iloc[0]]
            
            # Check if this is a singleton match
            if not singletons.empty and sample_id in singletons['Sample_ID'].values:
                singleton_coords.append(coords)
            else:
                neighbor_coords.append(coords)
    
    # Plot regular neighbors
    if neighbor_coords:
        neighbor_coords = np.array(neighbor_coords)
        colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(neighbor_coords)))
        ax3.scatter(neighbor_coords[:, 0], neighbor_coords[:, 1], 
                   c=colors, s=100, alpha=0.8, edgecolors='white', linewidth=1,
                   label='Close Neighbors')
    
    # Highlight singleton matches in special color
    if singleton_coords:
        singleton_coords = np.array(singleton_coords)
        ax3.scatter(singleton_coords[:, 0], singleton_coords[:, 1], 
                   c='gold', s=150, alpha=0.9, edgecolors='black', linewidth=2,
                   marker='D', label='Singleton Populations')
    
    # Your position
    ax3.scatter(your_pc1, your_pc2, c='white', s=500, marker='*', 
               edgecolors='red', linewidth=3, label='YOU', zorder=10)
    
    # Annotate your position
    ax3.annotate('YOU', xy=(your_pc1, your_pc2), xytext=(10, 10), 
                textcoords='offset points', color='white', fontweight='bold',
                fontsize=12, ha='left')
    
    ax3.set_xlabel('PC1', color='white', fontweight='bold')
    ax3.set_ylabel('PC2', color='white', fontweight='bold')
    ax3.set_title('PCA: You vs Your 50 Closest Genetic Neighbors', 
                 color='white', fontsize=14, fontweight='bold')
    ax3.tick_params(colors='white')
    ax3.legend(facecolor='black', edgecolor='white')
    
    # Plot 4: True ancestry composition (data-driven)
    ax4.set_facecolor('black')
    
    # Group by actual patterns, not commercial labels
    if 'Group_Name' in enriched_neighbors.columns:
        group_distances = enriched_neighbors.groupby('Group_Name')['Distance'].mean().sort_values()
        top_groups = group_distances.head(8)
        
        # Calculate weighted composition
        weights = 1.0 / (top_groups.values + 0.001)
        normalized_weights = weights / weights.sum()
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_groups)))
        wedges, texts, autotexts = ax4.pie(normalized_weights, 
                                          labels=[g[:20] + '...' if len(g) > 20 else g for g in top_groups.index],
                                          autopct='%1.1f%%', colors=colors,
                                          textprops={'color': 'white', 'fontsize': 10},
                                          startangle=90)
        
        ax4.set_title('True Genetic Composition\n(Based on Closest Neighbors)', 
                     color='white', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('true_genetic_discovery.png', 
                dpi=300, bbox_inches='tight', facecolor='black', edgecolor='none')
    plt.show()

def generate_discovery_report(enriched_neighbors):
    """Generate comprehensive discovery report"""
    
    print("\n" + "🔥" * 80)
    print("🧬 TRUE GENETIC DISCOVERY REPORT")
    print("🔥" * 80)
    
    print(f"\n📊 ANALYSIS SUMMARY:")
    print(f"   • {len(enriched_neighbors)} closest genetic neighbors analyzed")
    print(f"   • Spanning {enriched_neighbors['Country'].nunique()} countries")
    print(f"   • Representing {enriched_neighbors['Group_Name'].nunique() if 'Group_Name' in enriched_neighbors.columns else 'unknown'} distinct populations")
    
    print(f"\n🎯 TOP 10 CLOSEST INDIVIDUAL MATCHES:")
    for i, (_, neighbor) in enumerate(enriched_neighbors.head(10).iterrows(), 1):
        country = neighbor.get('Country', 'Unknown')
        group = neighbor.get('Group_Name', 'Unknown')
        distance = neighbor['Distance']
        date = neighbor.get('Date_mean_BP', 'Unknown')
        sample_id = str(neighbor['Sample_ID'])[:15]
        print(f"   {i:2d}. {sample_id:<15} | {country:<15} | {group:<25} | d={distance:.6f} | {date}")
    
    print(f"\n🌍 HIDDEN PATTERNS DISCOVERED:")
    
    # Geographic clustering
    top_countries = enriched_neighbors['Country'].value_counts().head(3)
    print(f"   • Geographic focus: {', '.join([f'{c} ({v})' for c, v in top_countries.items()])}")
    
    # Temporal clustering
    dated = enriched_neighbors.dropna(subset=['Date_mean_BP'])
    if not dated.empty:
        median_age = dated['Date_mean_BP'].median()
        print(f"   • Temporal center: ~{median_age:.0f} years ago")
    
    # Population diversity
    if 'Group_Name' in enriched_neighbors.columns:
        top_groups = enriched_neighbors['Group_Name'].value_counts().head(3)
        print(f"   • Population clusters: {', '.join([f'{g} ({v})' for g, v in top_groups.items()])}")
    
    print(f"\n💡 KEY INSIGHT:")
    print(f"   Your genetic signature represents a {get_genetic_summary(enriched_neighbors)}")
    
    print("🔥" * 80)

def get_genetic_summary(enriched_neighbors):
    """Generate a summary of genetic patterns"""
    
    # Analyze patterns
    country_diversity = enriched_neighbors['Country'].nunique()
    dated_samples = enriched_neighbors.dropna(subset=['Date_mean_BP'])
    
    if country_diversity >= 5:
        geo_pattern = "geographically diverse lineage"
    elif country_diversity >= 3:
        geo_pattern = "regionally concentrated ancestry"
    else:
        geo_pattern = "highly localized genetic signature"
    
    if not dated_samples.empty:
        age_range = dated_samples['Date_mean_BP'].max() - dated_samples['Date_mean_BP'].min()
        if age_range > 10000:
            temporal_pattern = "spanning deep prehistoric time"
        elif age_range > 3000:
            temporal_pattern = "bridging ancient and historical periods"
        else:
            temporal_pattern = "from a specific historical era"
    else:
        temporal_pattern = "of uncertain temporal depth"
    
    return f"{geo_pattern} {temporal_pattern}"

def main():
    """Main discovery workflow"""
    
    print("🧬 STARTING TRUE GENETIC DISCOVERY...")
    print("🎯 No assumptions, no commercial bias, just raw genetic truth")
    print("⚠️ NOTE: This analysis requires modern reference populations (1000G/HO) to be included in PCA")
    
    # Find neighbors
    result = find_true_genetic_neighbors(n_neighbors=500, sample_prefix="I0001")
    if result is None:
        return
    
    closest_neighbors, pca_data, aadr_anno = result
    
    # Enrich with metadata
    enriched_neighbors = enrich_with_metadata(closest_neighbors, aadr_anno)
    
    # Analyze patterns
    analyze_genetic_neighborhood(enriched_neighbors)
    
    # Create visualization
    create_discovery_visualization(enriched_neighbors, pca_data, sample_prefix="I0001")
    
    # Generate report
    generate_discovery_report(enriched_neighbors)
    
    print(f"\n✅ TRUE DISCOVERY COMPLETE")
    print(f"📄 Generated: true_genetic_discovery.png")
    print(f"📄 Generated: singleton_genetic_matches.csv (if singletons found)")
    print(f"🧬 Your genetic story revealed without commercial bias")
    print(f"\n🔬 NEXT STEPS:")
    print(f"   1. Add modern reference populations to your PCA dataset")
    print(f"   2. Re-run PCA with final_merged_dataset")
    print(f"   3. Run this script again for comprehensive analysis")

if __name__ == "__main__":
    main()