import torch
import numpy as np
import matplotlib.pyplot as plt
from isochrones import get_ichrone
from scipy import stats
import os,sys,pickle

from sbi import utils
from sbi import analysis
from sbi.inference.base import infer
from sbi.utils import process_prior

# First, some settings.
from sbi.inference import SNPE as method
from scipy import interpolate
import bz2
import _pickle as cPickle
observables = ['par','b','g','r','j','h','k','w1','w2']
(all_mags,all_errs,all_par_errs) = pickle.load(open('mag_err_data.pkl', 'rb'))
spls = []
for ii in range(len(observables)):
    inFile = open('../'+observables[ii]+'_err_data','r')
    mags = []
    errs = []
    bins = 30
    if ii==0:
        use_el = 1
    else:
        use_el = ii-1
    left = np.min(all_mags[:,use_el])
    right = np.max(all_mags[:,use_el])
    bin_width = (right-left)/bins
    for jj in range(bins):
        bin_left = left+jj*bin_width
        bin_right = left+(jj+1)*bin_width
        els = np.where((all_mags[:,use_el]>=bin_left) & (all_mags[:,use_el]<=bin_right))[0]
        if len(els)==0:
            continue
        if ii==0:
            use_errs = all_par_errs[els]
        else:
            use_errs = all_errs[els,use_el]
        mean_err = np.mean(use_errs)
        diffs = np.abs(use_errs-mean_err)
        mean_el = diffs.argmin()
        el = els[mean_el]
        mags.append(all_mags[el,use_el])
        if ii==0:
            errs.append(use_errs[mean_el]+0.07)
        else:
            errs.append(use_errs[mean_el])
    spl = interpolate.UnivariateSpline(mags,errs,s=10)
    spls.append(spl)

num_simulations = 150_000
num_samples = 2_000

labels = ("M1", "q", "age", "[M/H]", "log10(distance)")
short_labels = ("m1","q","age","feh","dist")
bounds = np.array([
    [0.4,  2.5], # M1
    [0,      1], # q
    [0.2,   10], # (Gyr)
    [-2, 0.5],  # metallicity
    [np.log10(50), np.log10(5000)] # log(distance)

])

from torch.distributions import (Uniform, Beta, Pareto, Independent)
from torch.distributions.transforms import AffineTransform
from torch.distributions.transformed_distribution import TransformedDistribution

from torch import tensor as tt

class StellarPrior:

    def __init__(
        self, 
        M1_bounds=(0.4, 5.0),   
        q_bounds=(0.0, 1.0),    
        tau_bounds=(1.0, 10.0), 
        m_h_bounds=(-2.0, 0.5), 
        distance_bounds=(1.0, 1000.0), 
        M1_alpha=1.0,
        M1_beta=5.0,
        m_h_alpha=10.0,
        m_h_beta=2.0,
        m_h_scale=3.0,
        return_numpy=False
    ):
        self.bounds = dict(
            lower_bound=tt([M1_bounds[0], q_bounds[0], tau_bounds[0], m_h_bounds[0], distance_bounds[0]]),
            upper_bound=tt([M1_bounds[1], q_bounds[1], tau_bounds[1], m_h_bounds[1], distance_bounds[1]])
        )
        self.lower = tt([M1_alpha, 1.0, 1.0, m_h_alpha, 1.0])
        self.upper = tt([M1_beta, 1.0, 1.0, m_h_beta, 1.0])
        m_h_mode = (m_h_alpha - 1)/(m_h_alpha + m_h_beta - 2)
        loc = tt([M1_bounds[0], q_bounds[0], tau_bounds[0], -m_h_mode * m_h_scale, distance_bounds[0]])
        scale = tt([M1_bounds[1], q_bounds[1], tau_bounds[1], m_h_scale, distance_bounds[1]])
        self.return_numpy = return_numpy
        self.dist = Independent(
            TransformedDistribution(
                Beta(self.lower, self.upper, validate_args=False),
                AffineTransform(loc=loc, scale=scale)
            ),
            1
        )

    def sample(self, sample_shape=torch.Size([])):
        samples = self.dist.sample(sample_shape)
        return samples.numpy() if self.return_numpy else samples

    def log_prob(self, values):
        if self.return_numpy:
            values = torch.as_tensor(values)
        log_probs = self.dist.log_prob(values)
        return log_probs.numpy() if self.return_numpy else log_probs



custom_prior = StellarPrior()
prior, *_ = process_prior(
    custom_prior,
    custom_prior_wrapper_kwargs=custom_prior.bounds
)

tracks = get_ichrone('mist', tracks=True)
def extinction(wavelength,distance):
	return 0.014*(wavelength/4.64e-7)**(-1.5)

def get_errors(all_app_mags):
    global spl
    all_errs = []
    for ii in range(len(spls)):
        if ii==0:
            in_mag = all_app_mags[1]
        else:
            in_mag = all_app_mags[ii-1]
        mode = spls[ii](in_mag)
        mode[np.where(mode<=0)] = 0.01
        sigma = 0.5
        mean = np.log(mode)+sigma**2
        error = np.random.lognormal(mean,sigma)
        all_errs.append(error)
    all_errs = np.array(all_errs)
    return all_errs

def get_mag_error(band,app_mag):
	if band=='b':
		coeffs1 = [-0.23793490460157124,4.225297710146195]
		coeffs2 = [-0.30737184372300963,7.661825650851712]
		sigma_0 = 0.00279017
	elif band=='g':
		coeffs1 = [-0.2031266476853316,4.28677633660234]
		coeffs2 = [-0.21957106489264594,6.924179835726976]
		sigma_0 = 0.0027553202
	elif band=='r':
		coeffs1 = [-0.2383368387390377,4.134722571200927]
		coeffs2 = [-0.29289902449149,7.483132861557515]
		sigma_0 = 0.0037793818
	minSNR = 10**(coeffs1[0]*app_mag+coeffs1[1])
	maxSNR = 10**(coeffs2[0]*app_mag+coeffs2[0])
	snr = minSNR+(maxSNR-minSNR)*np.random.random()
	mag_err = np.sqrt(((2.5/np.log(10))/snr)**2+sigma_0**2)
	return mag_err

def color_mag_no_errors(m1, q, age, fe_h, log_dist):
    properties = tracks.generate_binary(
        m1,
        q * m1,
        np.log10(age) + 9,
        fe_h,
        bands=["G", "BP", "RP", "J", "H", "K", "W1", "W2"]
    )
    all_abs_mags = np.array([properties.BP_mag.values,properties.G_mag.values,properties.RP_mag.values,properties.J_mag.values,properties.H_mag.values,properties.K_mag.values,properties.W1_mag.values,properties.W2_mag.values])
    #all_abs_mags = all_abs_mags[0:3]
    dist = np.float64(10**log_dist)
    all_app_mags = all_abs_mags+5*(np.float64(log_dist)-1)
    par = 1000/dist
    return_vals = [par]
    return_vals.extend(all_app_mags)
    return_vals = np.array(return_vals)
    return return_vals.T

def binary_color_mag_isochrones(m1, q, age, fe_h, log_dist):
    # isochrones.py needs log10(Age [yr]).
    # Our age is in Gyr, so we take log10(age * 10^9) = log10(age) + 9
    properties = tracks.generate_binary(
        m1,
        q * m1,
        np.log10(age) + 9,
        fe_h,
        bands=["G", "BP", "RP", "J", "H", "K", "W1", "W2"]
    )
    all_abs_mags = np.array([properties.BP_mag.values,properties.G_mag.values,properties.RP_mag.values,properties.J_mag.values,properties.H_mag.values,properties.K_mag.values,properties.W1_mag.values,properties.W2_mag.values])
    #all_abs_mags = all_abs_mags[0:3]
    dist = np.float64(10**log_dist)
    all_app_mags = all_abs_mags+5*(np.float64(log_dist)-1)
    mean_par = 1000/dist #parallax in mas
    all_errs = get_errors(all_app_mags)
    all_errs[np.where(np.isnan(all_errs))] = 0.1
    par_er = all_errs[0]
    if type(dist)==np.float64:
        n_par = 1
        #par_er = np.array([par_er])
        mean_par = np.array([mean_par])
    else:
        n_par = len(dist)
        #par_er = par_er*np.ones(n_par)
    par = np.random.normal(mean_par,par_er,n_par)
    while min(par)<0:
        n_neg = len(np.where(par<0)[0])
        par[np.where(par<0)] = np.random.normal(mean_par[np.where(par<0)],par_er[np.where(par<0)],n_neg)
    return_vals = [par]
    for ii in range(len(all_app_mags)):
        all_app_mags[ii] = np.random.normal(all_app_mags[ii],all_errs[ii+1],n_par)
        return_vals.append(all_app_mags[ii])
    return_vals.extend(all_errs)
    #print(return_vals)
    return_vals = np.array(return_vals)
    return return_vals.T

def simulator(theta):
    return torch.tensor(binary_color_mag_isochrones(*theta))

def model_mags(samples):
	properties = tracks.generate_binary(
        samples[:,0],
        0,
        np.log10(samples[:,2]) + 9,
        samples[:,3],
    	bands=["G", "BP", "RP", "J", "H", "K", "W1", "W2"]
    )
	all_mags = np.array([properties.BP_mag.values,properties.G_mag.values,properties.RP_mag.values,properties.J_mag.values,properties.H_mag.values,properties.K_mag.values,properties.W1_mag.values,properties.W2_mag.values])
	all_mags+=5*np.log10(samples[:,4])-5
	return all_mags

def chi2_calculate(samples,x):
	chi2s = np.zeros(samples.shape[0])
	for ii in range(len(chi2s)):
		chi2 = 0
		mags = model_mags(samples[ii])
		for jj in range(8):
			#print(mags[jj]>x_s[ii][jj+1])
			mags[jj][np.where(np.isnan(mags[jj]))[0]] = -20
			n_nan = len(np.where(np.isnan(mags[jj]))[0])
			err = x[ii][jj+10]
			chi2+=(np.sum(((mags[jj]>x[ii][jj+1])*(mags[jj]-x[ii][jj+1]))**2)/(err**2))/(mags.shape[0]*(mags.shape[1]-n_nan)-1)
		chi2s[ii] = chi2
	return chi2s
	
# Set priors.
from torch.distributions import (Uniform, Beta, Pareto)
from torch.distributions.transforms import AffineTransform
from torch.distributions.transformed_distribution import TransformedDistribution

from torch import tensor as tt

#prior = utils.BoxUniform(low=bounds.T[0], high=bounds.T[1])

from sbi.inference import prepare_for_sbi, simulate_for_sbi

#sbi_simulator, sbi_prior = prepare_for_sbi(simulator, prior)

#inference = method(prior)

# Generate the simulations. 
# We do this ourselves (instead of using simulate_for_sbi) because if we don't then many will be NaNs
# and we end up with fewer simulations than we want.
def simulate_for_sbi_strict(simulator, proposal, num_simulations, max_trials=np.inf):
    num_trials, num_simulated, theta, x = (0, 0, [], [])
    while num_simulated < num_simulations:
        N = num_simulations - num_simulated
        print(f"Running {N} simulations")
        _theta = proposal.sample((N, ))
        _x = simulator(_theta)
        #_theta, _x = simulate_for_sbi(simulator, proposal=proposal, num_simulations=N)
        keep = np.all(np.isfinite(_x).numpy(), axis=1)
        theta.extend(np.array(_theta[keep]))
        x.extend(np.array(_x[keep]))
        num_trials += 1
        num_simulated += sum(keep)
        if num_trials > max_trials:
            print(f"Warning: exceeding max trials ({max_trials}) with {num_simulated} / {num_simulations} simulations")
            break
    theta = torch.tensor(np.vstack(theta))
    x = torch.tensor(np.vstack(x))
    return (theta, x)

def find_extinction(A0,col):
	coeffs = [[1.153631975,-0.081401299,-0.036013024,0.019214359,-0.022397548,0.000840563,-1.31E-05,0.006601241,-0.000882248,-0.000111216],
	[0.995969722,-0.15972646,0.012238074,0.000907266,-0.037716026,0.001513475,-2.52E-05,0.011452266,-0.000936915,-0.000260297],
	[0.663207879,-0.017984716,0.000493769,-0.002679944,-0.006514221,3.30E-05,1.58E-06,-7.98E-05,0.00025568,1.10E-05],
	[0.340345411,-0.001502782,-0.000664573,0.000417668,-0.000156213,1.82E-07,2.34E-09,3.42E-06,-2.14E-07,4.79E-08],
	[0.255054351,0.000238094,-0.000948082,0.000286493,-6.84E-05,-2.41E-09,7.27E-10,1.88E-07,1.01E-06,1.12E-08],
	[0.19404852,-0.000259572,0.000495771,-0.000267507,-2.92E-05,8.02E-09,1.22E-10,6.47E-08,1.50E-07,2.70E-09]]
	bands = ['Bp','G','Rp','J','H','K']
	extinctions = np.zeros(len(bands))
	for ii in range(len(bands)):
		k_m = coeffs[ii][0]+coeffs[ii][1]*col+coeffs[ii][2]*col**2+coeffs[ii][3]*col**3+coeffs[ii][4]*A0+coeffs[ii][5]*A0**2+coeffs[ii][6]*A0**3+coeffs[ii][7]*col*A0+coeffs[ii][8]*(A0*col**2)+coeffs[ii][9]*(col*A0**2)
		extinctions[ii] = k_m*A0
	return extinctions
#sys.exit()
posterior = pickle.load(open('sbi_posterior_apparent.pkl', 'rb'))
file_name = 'gaia_data.csv'
with open(file_name, newline='') as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
		if '' in [row['parallax'],row['phot_bp_mean_mag'],row['phot_g_mean_mag'],row['phot_rp_mean_mag'],row['Jmag'],row['Kmag'],row['Hmag'],row['W1mag'],row['W2mag']]:
			continue
		if 'null' in [row['parallax'],row['phot_bp_mean_mag'],row['phot_g_mean_mag'],row['phot_rp_mean_mag'],row['Jmag'],row['Kmag'],row['Hmag'],row['W1mag'],row['W2mag']]:
			continue
		if '' in [row['azero_gspphot'],row['ebpminrp_gspphot']]:
			continue
		all_extinctions = find_extinction(float(row['azero_gspphot']),float(row['bp_rp']))
		A_G = all_extinctions[1]
		A_B = all_extinctions[0]
		A_R = all_extinctions[2]
		b_mag = float(row['phot_bp_mean_mag'])-A_B
		g_mag = float(row['phot_g_mean_mag'])-A_G
		r_mag = float(row['phot_rp_mean_mag'])-A_R
		if b_mag-r_mag<0:
			continue
		if g_mag-5*np.log10(100./float(row['parallax']))<4.3*(b_mag-r_mag)+0.12:
			continue
		A_J = all_extinctions[3]
		A_H = all_extinctions[4]
		A_K = all_extinctions[5]
		j_mag = float(row['Jmag'])-A_J
		h_mag = float(row['Hmag'])-A_H
		k_mag = float(row['Kmag'])-A_K
		data = [float(row['parallax']),b_mag,g_mag,r_mag,j_mag,h_mag,k_mag,float(row['W1mag']),float(row['W2mag'])]
		error_b = np.sqrt(((2.5/np.log(10))/float(row['phot_bp_mean_flux_over_error']))**2+0.0027901700**2)
		error_g = np.sqrt(((2.5/np.log(10))/float(row['phot_g_mean_flux_over_error']))**2+0.0027553202**2)
		error_r = np.sqrt(((2.5/np.log(10))/float(row['phot_rp_mean_flux_over_error']))**2+0.0037793818**2)
		if '' in [row['e_Jmag'],row['e_Hmag'],row['e_Kmag'],row['e_W1mag'],row['e_W2mag']]:
			continue
		if float(row['parallax'])<=0:
			continue
		dist = 1000/float(row['parallax'])
		errors = [float(row['parallax_error']),error_b,error_g,error_r,float(row['e_Jmag']),float(row['e_Hmag']),float(row['e_Kmag']),float(row['e_W1mag']),float(row['e_W2mag'])]
		data.extend(errors)
		all_data.append(data)
		all_ids.append(row['source_id'])
		coords = [float(row['ra']),float(row['ra_error']),float(row['dec']),float(row['dec_error'])]
		all_coords.append(coords)
all_data = np.array(all_data)
all_coords = np.array(all_coords)
pickle.dump((all_ids,all_data,all_coords), open('obs.pkl', 'wb'))
(all_ids,all_data,all_coords) = pickle.load(open('obs.pkl','rb'))
observation = all_data
from tqdm import tqdm
#_, L = sbi_prior.sample((1, )).shape
L = 5
saveDir = 'sample_save'
if not os.path.isdir(saveDir):
	os.system('mkdir '+saveDir)
#all_samples = []
nSplit = 100
place = 300
nPoints = len(observation)
#print(95*nPoints/nSplit)
#sys.exit()
for i, obs in enumerate(observation):
	section = i//(int(nPoints/nSplit))
	sample = posterior.sample(
	(num_samples,),
	x=obs,
	show_progress_bars=False
	)
	if sample.shape[0]<num_samples:
		sample = np.zeros((num_samples,L))
	percents = np.linspace(0,100,101)
	percentiles = np.percentile(sample,percents,axis=0)
	os.system('rm -rf current_*')
	textFile = open('current_'+str(i),'w')
	if i==int(section*nPoints/nSplit):
		all_percentiles = np.empty((len(percents), int(nPoints/nSplit), L))
		all_samples = np.empty((int(nPoints/nSplit), num_samples, L))
	all_percentiles[:,i-section*int(nPoints/nSplit)] = percentiles
	all_samples[i-section*int(nPoints/nSplit)] = sample
	if (i+1)//(int(nPoints/nSplit))==section+1:#(i+1)%(nPoints/nSplit)==0:
		start = i+1-int(nPoints/nSplit)
		end = i+1
		pickle.dump((np.array(all_ids[start:end]),observation[start:end],np.array(all_percentiles)), open(saveDir+'/sbi_percentiles_'+str(int(place+section))+'.pkl', 'wb'))
		bs = chi2_calculate(all_samples,observation[start:end])
		pickle.dump((np.array(all_ids[start:end]),observation[start:end],all_coords[start:end],np.array(all_percentiles)[50],bs),open(saveDir+'/bs_'+str(int(place+section))+'.pkl','wb'))
		with bz2.BZ2File(saveDir+'/sbi_samples_'+str(int(place+section))+'.pbz2','w') as f: 
			pickle.dump((np.array(all_ids[start:end]),observation[start:end],all_coords[start:end],np.array(all_samples)), f)