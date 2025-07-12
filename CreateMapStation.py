from matplotlib.pyplot import figure, plot, show
from matplotlib import rcParams
from mpl_toolkits.basemap import Basemap
from numpy import arange

rcParams["font.size"] = 16
rcParams["savefig.dpi"] = 400
Figure = figure(1, figsize=(12, 6))
MexMap = Basemap(projection="cyl", resolution="h",
              llcrnrlat=14, urcrnrlat=34,
              llcrnrlon=-118, urcrnrlon=-86)

MexMap.shadedrelief()
MexMap.drawcoastlines()
MexMap.drawcountries()
LinesParallels = arange(14, 34, 4)
LinesMeridians = arange(-78, -118, -8)
MexMap.drawparallels(LinesParallels, labels=[True, False, False, False])
MexMap.drawmeridians(LinesMeridians, labels=[False, False, True, False])

ptex_coords = (-116.52124, 32.28845)
ptex_color = "red"

ptex_coords_in_map = MexMap(*ptex_coords)
plot(*ptex_coords_in_map, "^", color=ptex_color, markersize = 12, label="PTEX")

Figure.legend(loc=7, fancybox=True, shadow=True)
show()
