# LiBs_leaching

Prediction of metal leaching from LiBs using ML.

The database contains 6643 entries relative to the leaching of metals from spent LiBs (Cu and Al current collectors, Fe from casing, Co, Ni, Mn, and Li from black mass).

We were willing to compare the results to the work published in the Journal of Cleaner Production, Volume 410, 15 July 2023, 137188. The article titled "Application of machine learning to guide efficient metal leaching from spent lithium-ion batteries and comprehensively reveal the process parameter influences" by Bo Niu, Xiaomin Wang, and Zhenming Xu provides a comprehensive analysis of the leaching process using machine learning (10.1016/j.jclepro.2023.137188). This repository will allow one to train its own model and compare with the results that could be obtained.

### Database Description

The database includes the following features and targets (metals leaching efficiency):

- Thermal treatment (°C)
- Li in feed (%)
- Co in feed (%)
- Mn in feed (%)
- Al in feed (%)
- Ni in feed (%)
- Fe in feed (%)
- Cu in feed (%)
- Solid to liquid ratio (g/L)
- Particle size (µm)
- Reductant concentration (M)
- Standard Reduction potential of the reducing agent
- Acid concentration (M)
- Acid pKa
- Stirring speed (rpm)
- Leaching temperature (°C)
- Mixing time (min)
- Li leached (%)
- Co leached (%)
- Mn leached (%)
- Al leached (%)
- Ni leached (%)
- Fe leached (%)
- Cu leached (%)
- Reference (DOI): paper from which the data was obtained

Other sheets in the Excel database describe the acids included with their pKa, the reductant redox potential, and the size of the cations.

We have trained and compared multiple ML algorithms and calculated the associated statistics, which allow a proper comparison of the efficiency of various algorithms.
