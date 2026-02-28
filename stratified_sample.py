import pandas as pd
import numpy as np

# Read the data
df = pd.read_csv('nacti_metadata.csv')

# Get animal types and their counts
animal_counts = df['common_name'].value_counts()
total_images = len(df)
elk_count = animal_counts.get('elk', 0)

print(f'Total number of images: {total_images}')
print(f'Original animal distribution:')
print(animal_counts)
print()

# Define sample sizes
elk_sample_size = 22000  # Elk: ~22000 photos
other_animals_sample_size = 22000  # Other animals: ~22000 photos
sample_size = elk_sample_size + other_animals_sample_size

# Get non-elk animals and their distribution
non_elk_df = df[df['common_name'] != 'elk']
non_elk_counts = non_elk_df['common_name'].value_counts()
non_elk_total = len(non_elk_df)

# Create stratified sample
stratified_df = pd.DataFrame()

# Sample elk (50%)
if elk_count > 0:
    elk_df = df[df['common_name'] == 'elk']
    elk_sample = elk_df.sample(n=min(elk_sample_size, len(elk_df)), random_state=42)
    stratified_df = pd.concat([stratified_df, elk_sample])

# Sample other animals proportionally (50%)
for animal in non_elk_counts.index:
    animal_df = df[df['common_name'] == animal]
    # Calculate proportion of this animal in non-elk animals
    proportion = len(animal_df) / non_elk_total
    # Calculate how many samples this animal should get from the 50% non-elk budget
    sample_count = int(other_animals_sample_size * proportion)
    if sample_count > 0 and len(animal_df) > 0:
        animal_sample = animal_df.sample(n=min(sample_count, len(animal_df)), random_state=42)
        stratified_df = pd.concat([stratified_df, animal_sample])

# Reset index
stratified_df = stratified_df.reset_index(drop=True)

# Save the stratified sample
stratified_df.to_csv('stratified_sample.csv', index=False)

print(f'Stratified sample size: {len(stratified_df)}')
print(f'Stratified sample distribution:')
print(stratified_df['common_name'].value_counts())
print()
print(f'Stratified sample proportions:')
print(stratified_df['common_name'].value_counts(normalize=True))
print()
print("Stratified sample saved to 'stratified_sample.csv'")
