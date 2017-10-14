/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 1000;

	// Create normal distributions for x, y and theta.
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	default_random_engine gen;

	for (int i = 0; i < num_particles; ++i) {
		double sample_x, sample_y, sample_theta;

		// Sample from these normal distrubtions
		sample_x = dist_x(gen);
		sample_y = dist_y(gen);
		sample_theta = dist_theta(gen);

		Particle p;
		p.id = i;
		p.x = sample_x;
		p.y = sample_y;
		p.theta = sample_theta;
		p.weight = 1;
		particles.push_back(p);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	default_random_engine gen;

	// Predict new position based on odometry
	for (int i = 0; i < num_particles; ++i) {

		if (fabs(yaw_rate) > 0.00001) {
			particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
			particles[i].theta += yaw_rate * delta_t;
		}
		else {
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}

		// Add noise
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	// Find closest from observation to predicted point
	for (int i = 0; i < observations.size(); i++)
	{
		// init minimum distance to maximum possible and id to -1
		double min_dist = numeric_limits<double>::max();
		int min_id = -1;

		// Loop through all possible predictions and assign the one with the closest distance
		for (int j = 0; j < predicted.size(); j++)
		{
			double curr_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

			if (curr_dist < min_dist)
			{
				min_id = predicted[j].id;
				min_dist = curr_dist;
			}	
		}

		// Update observation ID to the ID with least amount of distance
		observations[i].id = min_id;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// Iterate through each particle and update the observation in relation to its coordinates
	for (int i = 0; i < num_particles; ++i)
	{
		// Iterate through each landmark and calculate which landmark is within the sensor range
		vector<LandmarkObs> landmarks_in_range;
		for (int j = 0;j < map_landmarks.landmark_list.size(); ++j)
		{
			if (fabs(particles[i].x - map_landmarks.landmark_list[j].x_f) < sensor_range && fabs(particles[i].y - map_landmarks.landmark_list[j].y_f) < sensor_range)
			{
				landmarks_in_range.push_back(LandmarkObs{ map_landmarks.landmark_list[j].id_i,map_landmarks.landmark_list[j].x_f,map_landmarks.landmark_list[j].y_f});
			}
		}

		// Need to map each observation from vehicle coordinates to map coordinates
		vector<LandmarkObs> trans_observation;
		for (int j = 0;j < observations.size(); ++j)
		{
			double x_map = particles[i].x + (cos(particles[i].theta) * observations[j].x) - (sin(particles[i].theta) * observations[j].y);
			double y_map = particles[i].y + (sin(particles[i].theta) * observations[j].x) + (cos(particles[i].theta) * observations[j].y);
			trans_observation.push_back(LandmarkObs{ observations[j].id, x_map, y_map });
		}
		
		// Perform association with landmarks_in_range
		dataAssociation(trans_observation, landmarks_in_range);

		// Update weights
		// Loop through all transformed observations and update their weights
		for (int j = 0;j < trans_observation.size(); ++j)
		{
			double x_obs = trans_observation[j].x;
			double y_obs = trans_observation[j].y;
			double mu_x = 0;
			double mu_y = 0;
			double sig_x = std_landmark[0];
			double sig_y = std_landmark[1];

			for (int k = 0;k < landmarks_in_range.size(); ++k)
			{
				if (trans_observation[j].id == landmarks_in_range[k].id)
				{
					mu_x = landmarks_in_range[k].x;
					mu_y = landmarks_in_range[k].y;
					break;
				}
			}
			// Calculate normalization term
			double gauss_norm = (1 / (2 * M_PI * sig_x * sig_y));

			// Calculate exponent
			double exponent = (pow((x_obs - mu_x),2)) / (2.0 * pow(sig_x,2)) + (pow((y_obs - mu_y),2)) / (2 * pow(sig_y,2));

			// Calculate weight using normalization terms and exponent - Multivariate-Gaussian Probability
			double weight = gauss_norm * exp(-exponent);

			// Update particles final weight
			particles[i].weight *= weight;

		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
