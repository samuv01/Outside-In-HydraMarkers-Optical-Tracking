import numpy as np
from scipy import stats
from scipy.spatial.transform import Rotation

def analyze_translation_jitter_from_data(translations_mm, confidence=0.95):
    """Analyze translational jitter with chi-squared confidence interval."""
    n = translations_mm.shape[0]
    alpha = 1 - confidence
    
    mean_pos = np.mean(translations_mm, axis=0)
    deviations = np.linalg.norm(translations_mm - mean_pos, axis=1)
    
    sample_var = np.var(deviations, ddof=1)
    sample_std = np.std(deviations, ddof=1)
    
    chi2_lower = stats.chi2.ppf(alpha/2, n-1)
    chi2_upper = stats.chi2.ppf(1 - alpha/2, n-1)
    
    var_ci_lower = (n - 1) * sample_var / chi2_upper
    var_ci_upper = (n - 1) * sample_var / chi2_lower
    
    std_ci_lower = np.sqrt(var_ci_lower)
    std_ci_upper = np.sqrt(var_ci_upper)
    
    return {
        'mean_position': mean_pos,
        'mean_deviation': np.mean(deviations),
        'sample_variance': sample_var,
        'sample_std': sample_std,
        'variance_ci_lower': var_ci_lower,
        'variance_ci_upper': var_ci_upper,
        'std_ci_lower': std_ci_lower,
        'std_ci_upper': std_ci_upper,
        'n_samples': n,
        'alpha': alpha
    }

def analyze_rotation_jitter_from_matrices(rotation_matrices, confidence=0.95):
    """Analyze rotational jitter from rotation matrices."""
    n = rotation_matrices.shape[0]
    alpha = 1 - confidence
    
    # Convert to Rotation objects
    rotations = [Rotation.from_matrix(R) for R in rotation_matrices]
    
    # Mean rotation
    mean_matrix = np.mean(rotation_matrices, axis=0)
    mean_rotation = Rotation.from_matrix(mean_matrix)
    
    # Compute deviations
    rotation_deviations_rad = []
    for R_frame in rotations:
        delta_R = R_frame * mean_rotation.inv()
        angle_rad = np.linalg.norm(delta_R.as_rotvec())
        rotation_deviations_rad.append(angle_rad)
    
    rotation_deviations_rad = np.array(rotation_deviations_rad)
    rotation_deviations_deg = np.degrees(rotation_deviations_rad)
    
    sample_var_rad = np.var(rotation_deviations_rad, ddof=1)
    sample_std_rad = np.std(rotation_deviations_rad, ddof=1)
    sample_std_deg = np.degrees(sample_std_rad)
    
    chi2_lower = stats.chi2.ppf(alpha/2, n-1)
    chi2_upper = stats.chi2.ppf(1 - alpha/2, n-1)
    
    var_ci_lower_rad = (n - 1) * sample_var_rad / chi2_upper
    var_ci_upper_rad = (n - 1) * sample_var_rad / chi2_lower
    
    std_ci_lower_rad = np.sqrt(var_ci_lower_rad)
    std_ci_upper_rad = np.sqrt(var_ci_upper_rad)
    
    std_ci_lower_deg = np.degrees(std_ci_lower_rad)
    std_ci_upper_deg = np.degrees(std_ci_upper_rad)
    
    return {
        'mean_deviation_rad': np.mean(rotation_deviations_rad),
        'mean_deviation_deg': np.degrees(np.mean(rotation_deviations_rad)),
        'sample_variance_rad': sample_var_rad,
        'sample_std_rad': sample_std_rad,
        'sample_std_deg': sample_std_deg,
        'variance_ci_lower_rad': var_ci_lower_rad,
        'variance_ci_upper_rad': var_ci_upper_rad,
        'std_ci_lower_deg': std_ci_lower_deg,
        'std_ci_upper_deg': std_ci_upper_deg,
        'n_samples': n,
        'alpha': alpha
    }

path = r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\6x10_3x3tag_1x10tag_6x2tag_bis\jitter\trial_01\tracking_results.npy"
data = np.load(path, allow_pickle=True).item()

translations_mm = np.asarray(data["translations_mm"])
rotation_matrices = np.asarray(data["rotation_matrices"])

trans_jitter = analyze_translation_jitter_from_data(translations_mm, confidence=0.95)

print("=" * 70)
print("TRANSLATIONAL JITTER ANALYSIS (95% CI)")
print("=" * 70)
print(f"Number of frames: {trans_jitter['n_samples']}")
print(f"Mean position (mm): {trans_jitter['mean_position']}")
print()
print(f"Mean deviation from center: {trans_jitter['mean_deviation']:.6f} mm")
print(f"Sample Std Dev: {trans_jitter['sample_std']:.6f} mm")
print(f"  95% CI: [{trans_jitter['std_ci_lower']:.6f}, {trans_jitter['std_ci_upper']:.6f}] mm")
print()

rot_jitter = analyze_rotation_jitter_from_matrices(rotation_matrices, confidence=0.95)

print("=" * 70)
print("ROTATIONAL JITTER ANALYSIS (95% CI)")
print("=" * 70)
print(f"Number of frames: {rot_jitter['n_samples']}")
print()
print(f"Mean deviation: {rot_jitter['mean_deviation_deg']:.6f}°")
print(f"Sample Std Dev: {rot_jitter['sample_std_deg']:.6f}°")
print(f"  95% CI: [{rot_jitter['std_ci_lower_deg']:.6f}°, {rot_jitter['std_ci_upper_deg']:.6f}°]")
print()
