# Introduction
Takes in root file of delphes events and outputs a csv file. The csv file contains
# Explanation of Terms
## Delphes Data Attributes
| var name        	| branch      	| subbranch         	| description                  	|
|-----------------	|-------------	|-------------------	|------------------------------	|
| l_genPID        	| 'Particle'  	| 'Particle.PID'    	| unique event ID              	|
| l_genStatus     	| 'Particle'  	| 'Particle.Status' 	|                              	|
| l_genPt         	| 'Particle'  	| 'Particle<span></span>.PT'     	| particle transverse momentum 	|
| l_genEta        	| 'Particle'  	| 'Particle.Eta'    	| particle Eta                 	|
| l_genPhi        	| 'Particle'  	| 'Particle.Phi'    	| Particle Phi                          	|
| l_genMass       	| 'Particle'  	| 'Particle.Mass'   	| particle mass                	|
| l_jetPt         	| 'Jet'       	| 'Jet.<span></span>PT'          	| jet transverse momentum      	|
| l_jetEta        	| 'Jet'       	| 'Jet.Eta'         	| jet Eta                      	|
| l_jetPhi        	| 'Jet'       	| 'Jet.Phi'         	| jet Phi                      	|
| l_jetMass       	| 'Jet'       	| 'Jet.Mass'        	| jet Mass                     	|
| l_jetBTag       	| 'Jet'       	| 'Jet.BTag'        	| whether or not it is a b quark jet 	|
| l_missingET_met 	| 'MissingET' 	| 'MissingET.MET'   	|                              	|
| l_missingET_Phi 	| 'MissingET' 	| 'MissingET.Phi'   	|                              	|
| l_scalarHT      	| 'ScalarHT'  	| 'ScalarHT.<span></span>HT'     	|                              	|

### Particle

### Jet

### Missing ET

### Scalar HT

## Functions for Plotting
### plotOneHistogram
Takes in data, figure number, title, x label, min and max values, and number of
bins and creaates histogram.
### compareManyHisograms