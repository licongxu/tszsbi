# load_samples.jl
using CSV
using DataFrames
using Plots
using XGPaint
using Healpix
using Pixell

# Include the custom GNFW profile definitions.
# include("profile.jl")

# --- (Optional) If you want to use your custom profile with updated parameters, instantiate it:
# Use your desired parameters:
# h: 0.68,
# Ob0h2: 0.049*0.68^2,  Oc0h2: (0.31-0.049)*0.68^2  ⇒ Ωₘ = 0.31,
# Profile parameters:
# P0GNFW: 8.130, c500: 1.156, gammaGNFW: 0.3292, alphaGNFW: 1.0620, betaGNFW: 5.4807, B: 1.2
# y_model_interp = CustomGNFWProfile(H0=68.0, B=1.2, c500=1.156, gammaGNFW=0.3292,
#                                    alphaGNFW=1.0620, betaGNFW=5.4807, P0GNFW=8.130)
# println("Using custom GNFW profile: ", y_model_interp)

# --- Read the CSV file into a DataFrame
file_path = "/Users/licongxu/Work/catalogues/catalogue_so_1e14_0.csv"
df = CSV.read(file_path, DataFrame)

# Extract the columns from the DataFrame
redshift = df.z
halo_mass = df.M .* 1e14  # Convert from units of 1e14 M_sun to M_sun
lon = df.lon
lat = df.lat

# Convert the stored sky coordinates:
#   - Convert 'lon' (originally in [0, 2π]) to right ascension in [-π, π]
#   - Convert the polar angle 'lat' to declination via dec = π/2 - lat
ra = rem.(lon .+ π, 2π) .- π
dec = (π/2) .- lat

println("Number of halos: ", length(halo_mass))

# Optionally, sort the halo catalog (assuming sort_halo_catalog is available from XGPaint)
ra, dec, redshift, halo_mass = sort_halo_catalog(ra, dec, redshift, halo_mass);
y_model_interp = XGPaint.load_precomputed_battaglia()
print(y_model_interp)

# HEALPix Map Processing
nside = 512
m_hp = HealpixMap{Float64,RingOrder}(nside)
max_radius = deg2rad(5.0)  # Maximum radius to consider for the profile
w = HealpixProfileWorkspace(nside, max_radius)

# Paint the HEALPix map
@time paint!(m_hp, w, y_model_interp, halo_mass, redshift, ra, dec)

# Save the HEALPix map as a PNG image
q = plot(m_hp)
savefig(q, "/Users/licongxu/Work/fits_maps_cat/tsz_map_hp.png")  
println("Saved plot tsz_map_hp.png")

# Save the HEALPix map as a FITS file
fits_file_path = "/Users/licongxu/Work/fits_maps_cat/tsz_map_hp.fits"
Healpix.saveToFITS(m_hp, fits_file_path, typechar="D")  
println("Saved HEALPix map as FITS: $fits_file_path")
