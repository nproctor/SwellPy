import ParticleSuspension as ps

def main():
	# create a monodisperse system with 1000 particles
	md = ps.monodisperse(1000)
	# find the diameter that corresponds to 80% area fraction
	af80 = md.percent_to_diameter(.8)
	# train on this diameter with a "kick" of 1/N (= 1/1000)
	md.train_for(af80, (1/1000), 100)
	# plot the particles with extended periodic boundaries
	md.particle_plot(af80, extend = True)
	# plot tag count
	md.tag_plot(mode = 'count')
	# plot tag rate
	md.tag_plot(mode = 'rate')
	# plot the 2nd derivative of the tag count
	md.tag_plot(mode = 'curve')


if __name__ == "__main__":
	main()