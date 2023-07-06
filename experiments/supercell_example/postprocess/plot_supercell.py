#!/usr/bin/env python3

from netCDF4 import Dataset
import argparse
from math import pi, sqrt, cos, sin, atan2, nan
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pathlib
# from matplotlib.gridspec import GridSpec

def get_parser():
  """
    create a command line parser for this program
  """
  parser = argparse.ArgumentParser(
    prog="plot_supercell.py",
    description="plot output from miniWeatherML's supercell example." #,
    #epilog=
    )
  parser.add_argument("input_file",
    help="path to netcdf output data file from the supercell experiment.",
    metavar="NC_DATA.nc")
  parser.add_argument("-y",
    "--y_idx",
    type=int,
    default=0,
    help="constant y index for 2D x-z profiles.")
  parser.add_argument("-T",
    "--temperature",
    action="store_true",
    help="plot temperature")
  parser.add_argument("-u",
    "--horizontal_x_velocity",
    action="store_true",
    help="plot horizontal x velocity")
  parser.add_argument("-v",
    "--horizontal_y_velocity",
    action="store_true",
    help="plot horizontal y velocity")
  parser.add_argument("-w",
    "--vertical_velocity",
    action="store_true",
    help="plot vertical velocity")
  parser.add_argument("-qv",
    "--water_vapor",
    action="store_true",
    help="plot water vapor")
  parser.add_argument("-qc",
    "--cloud_liquid",
    action="store_true",
    help="plot cloud liquid")
  parser.add_argument("-qi",
    "--cloud_ice",
    action="store_true",
    help="plot cloud ice")
  parser.add_argument("-qr",
    "--rain",
    action="store_true",
    help="plot rain")
  parser.add_argument("-rho",
    "--density",
    action="store_true",
    help="plot density")
  parser.add_argument("-t",
    "--start_idx",
    type=int,
    default=0,
    help="start time index to plot")
#   parser.add_argument("-s",
#     "--stop_idx",
#     type=int,
#     default=0,
#     help="plot frames for all time indices in [0, stop_idx)")
  parser.add_argument("-V",
    "--verbose",
    action="store_true",
    help="write extra info to console while processing.")
  parser.add_argument("-o",
    "--output_file",
    default="supercell_plots.png",
    help="output filename (must be a .png file)")
  parser.add_argument("-a",
    "--plot-all",
    action="store_true",
    help="plot all output variables.")
  return parser

def default_plot_order():
  """
    return a list indicating the default plotting order.
  """
  result = ["vertical_velocity",
            "temperature",
            "water_vapor",
            "cloud_liquid",
            "cloud_ice",
            "rain",
            "horizontal_x_velocity",
            "horizontal_y_velocity",
            "density"]
  return result

def count_plots(args):
  """
    Count the number of variables to be plotted in each frame.
    Return a mask such that mask[i] = True if variable i will be plotted.
    The indices correspond to the default plot ordering.
  """
  if args.plot_all:
    ct = 9
    mask = [True for i in range(9)]
  else:
    ct = 0
    mask = [False for i in range(9)]
    porder = default_plot_order()
    if args.temperature:
      result += 1
      mask[porder["temperature"]] = True
      if args.verbose:
        print("  plotting T")
    if args.horizontal_x_velocity:
      result += 1
      mask[porder["horizontal_x_velocity"]] = True
      if args.verbose:
        print("  plotting u")
    if args.horizontal_y_velocity:
      result += 1
      mask[porder["horizontal_y_velocity"]] = True
      if args.verbose:
        print("  plotting v")
    if args.vertical_velocity:
      result += 1
      mask[porder["vertical_velocity"]] = True
      if args.verbose:
        print("  plotting w")
    if args.water_vapor:
      result += 1
      mask[porder["water_vapor"]] = True
      if args.verbose:
        print("  plotting qv")
    if args.cloud_liquid:
      result += 1
      mask[porder["cloud_liquid"]] = True
      if args.verbose:
        print("  plotting qc")
    if args.cloud_ice:
      result += 1
      mask[porder["cloud_ice"]] = True
      if args.verbose:
        print("  plotting qi")
    if args.rain:
      result += 1
      mask[porder["rain"]] = True
      if args.verbose:
        print("  plotting qr")
    if args.density:
      result += 1
      mask[porder["density"]] = True
      if args.verbose:
        print("  plotting rho")
  return ct, mask

def get_subplot_layout(np):
  """
    Determine the subplot layout given the number of variables to plot.
  """
  if np == 1:
    nrow = 1
    ncol = 1
  elif np <= 5:
    nrow = np
    ncol = 1
  elif np <= 8:
    nrow = np/2
    ncol = 2
  else:
    nrow = 3
    ncol = 3
  return (nrow, ncol)

def get_variable_axes(args):
  np, mask = count_plots(args)
  (nrow, ncol) = get_subplot_layout(np)
  if args.plot_all:
    ax_idx = {"vertical_velocity":(0,0),
     "temperature":(0,1),
     "water_vapor":(0,2),
     "cloud_liquid":(1,0),
     "cloud_ice":(1,1),
     "rain":(1,2),
     "horizontal_x_velocity":(2,0),
     "horizontal_y_velocity":(2,1),
     "density":(2,2)}
  return ax_idx

def check_args(args):
  """
    detect invalid input arguments and output error if when found.
  """
  nerr = 0
  if pathlib.Path(args.input_file).suffix != ".nc":
    print("error: expected a .nc input file (received {})".format(args.input_file))
    nerr += 1
  if pathlib.Path(args.output_file).suffix != ".png":
    print("error: output must be a .png file (received {})".format(args.output_file))
    nerr += 1
  np, m = count_plots(args)
  if np == 0:
    print("error: 0 plots requested.")
    nerr += 1
#   if args.stop_idx > 0 and args.stop_idx < args.start_idx:
#     print("error: timne stop index must be greater than or equal to time start index")
#     nerr += 1
  if nerr > 0:
    raise ValueError("invalid argument(s) found. Re-run with --help for usage.")

def plot_frame(nrow, ncol, plot_mask, axs, t_idx, ncd):
  pass

def plot_vertical_velocity(ax, t_idx, y_idx, ncd):
  x = ncd.variables["x"][:]
  z = ncd.variables["z"][:]
  w = ncd.variables["wvel"][t_idx, :, y_idx, :]
  levels = np.linspace(-10,10)
  cp = ax.contourf(x, z, w, levels, cmap=mpl.colormaps["PuOr_r"])
  cticks = np.linspace(-10,10,5)
  cbar = plt.colorbar(cp, ax=ax, ticks=cticks)
  ax.set_title("w [m/s]")

def plot_horizontal_x_velocity(ax, t_idx, y_idx, ncd):
  x = ncd.variables["x"][:]
  z = ncd.variables["z"][:]
  u = ncd.variables["uvel"][t_idx, :, y_idx, :]
  levels = np.linspace(-30,30)
  cticks = np.linspace(-30,30,5)
  cp = ax.contourf(x, z, u, levels, cmap=mpl.colormaps["PiYG"])
  plt.colorbar(cp, ax=ax, ticks=cticks)
  ax.set_title("u [m/s]")

def plot_horizontal_y_velocity(ax, t_idx, y_idx, ncd):
  x = ncd.variables["x"][:]
  z = ncd.variables["z"][:]
  v = ncd.variables["vvel"][t_idx, :, y_idx, :]
  levels = np.linspace(-30,30)
  cticks = np.linspace(-30,30,5)
  cp = ax.contourf(x, z, v, cmap=mpl.colormaps["PiYG"])
  plt.colorbar(cp, ax=ax, ticks=cticks)
  ax.set_title("v [m/s]")

def plot_temperature(ax, t_idx, y_idx, ncd):
  x = ncd.variables["x"][:]
  z = ncd.variables["z"][:]
  T = ncd.variables["temp"][t_idx, :, y_idx, :]
  levels = np.linspace(200,320)
  cticks = np.linspace(200,320,7)
  cp = ax.contourf(x, z, T, levels, cmap=mpl.colormaps["Spectral_r"])
  plt.colorbar(cp, ax=ax,ticks=cticks)
  ax.set_title("T [K]")

def plot_density(ax, t_idx, y_idx, ncd):
  x = ncd.variables["x"][:]
  z = ncd.variables["z"][:]
  rho = ncd.variables["density_dry"][t_idx, :, y_idx, :]
  levels = np.linspace(0,1.5)
  cticks = np.linspace(0,1.5,5)
  cp = ax.contourf(x, z, rho, levels, cmap=mpl.colormaps["gist_yarg"])
  plt.colorbar(cp, ax=ax, ticks=cticks)
#   ax.set_title("$\rho_{\text{dry}}$")
  ax.set_title("rho_dry [kg/m3]")

def plot_cloud_ice(ax, t_idx, y_idx, ncd):
  if "cloud_ice" in ncd.variables:
    x = ncd.variables["x"][:]
    z = ncd.variables["z"][:]
    qi = ncd.variables["cloud_ice"][t_idx, :, y_idx, :]
#   levels = np.linspace(0, 0.02)
    cp = ax.contourf(x, z, qi, cmap=mpl.colormaps["Purples"])
    cp.cmap.set_under("red")
    cp.cmap.set_over("red")
    plt.colorbar(cp, ax=ax)
  ax.set_title("qi [kg / kg]")

def plot_water_vapor(ax, t_idx, y_idx, ncd):
  x = ncd.variables["x"][:]
  z = ncd.variables["z"][:]
  qv = ncd.variables["water_vapor"][t_idx, :, y_idx, :]
  levels = np.linspace(0, 0.02)
  cticks = np.linspace(0, 0.02, 5)
  cp = ax.contourf(x, z, qv, levels, cmap=mpl.colormaps["Blues"])
  plt.colorbar(cp, ax=ax, ticks=cticks)
  ax.set_title("qv [kg / kg]")

def plot_cloud_liquid(ax, t_idx, y_idx, ncd):
  x = ncd.variables["x"][:]
  z = ncd.variables["z"][:]
  qc = ncd.variables["cloud_liquid"][t_idx, :, y_idx, :]
  levels = np.linspace(0, 0.002)
  cticks = np.linspace(0, 0.002, 5)
  cp = ax.contourf(x, z, qc, levels, cmap=mpl.colormaps["Greens"])
  cp.cmap.set_under("red")
  cp.cmap.set_over("red")
  plt.colorbar(cp, ax=ax, ticks=cticks)
  ax.set_title("qc [kg / kg]")

def plot_rain(ax, t_idx, y_idx, ncd):
  x = ncd.variables["x"][:]
  z = ncd.variables["z"][:]
  qr = ncd.variables["precip_liquid"][t_idx, :, y_idx, :]
  levels = np.linspace(0, 0.003)
  cticks = np.linspace(0, 0.003, 6)
  cp = ax.contourf(x, z, qr, cmap=mpl.colormaps["PuRd"])
  cp.cmap.set_under("red")
  cp.cmap.set_over("red")
  plt.colorbar(cp, ax=ax, ticks=cticks)
  ax.set_title("qr [kg / kg]")

if __name__ == "__main__":
  parser = get_parser()
  args = parser.parse_args()

  args.verbose = True
  args.plot_all = True
  check_args(args)

  ncd = Dataset(args.input_file, "r")

  # TODO: fix this to plot subsets of variables
  nplots, plot_mask = count_plots(args)
  if args.verbose:
    print("setting up {} subplots per frame...".format(nplots))

  nplots = 9
  (nrow, ncol) = get_subplot_layout(nplots)

  if args.verbose:
    print("  subplot layout has {} rows and {} columns.".format(nrow, ncol))
  fig, axs = plt.subplots(nrows=nrow, ncols=ncol, sharex=True, sharey=True,layout="constrained")

  t_idx = 38
  y_idx = args.y_idx
  varnames = default_plot_order()
  ax_idx = get_variable_axes(args)
  for v in varnames:
    i, j = ax_idx[v]
    if v == "vertical_velocity":
      plot_vertical_velocity(axs[i,j], t_idx, y_idx, ncd)
    elif v == "temperature":
      plot_temperature(axs[i,j], t_idx, y_idx, ncd)
    elif v == "water_vapor":
      plot_water_vapor(axs[i,j], t_idx, y_idx, ncd)
    elif v == "cloud_liquid":
        plot_cloud_liquid(axs[i,j], t_idx, y_idx, ncd)
    elif v == "cloud_ice":
        plot_cloud_ice(axs[i,j], t_idx, y_idx, ncd)
    elif v == "rain":
      plot_rain(axs[i,j], t_idx, y_idx, ncd)
    elif v == "horizontal_x_velocity":
      plot_horizontal_x_velocity(axs[i,j], t_idx, y_idx, ncd)
    elif v == "horizontal_y_velocity":
      plot_horizontal_y_velocity(axs[i,j], t_idx, y_idx, ncd)
    elif v == "density":
      plot_density(axs[i,j], t_idx, y_idx, ncd)

  for j in range(ncol):
    x = ncd.variables["x"][:]
    minx = np.min(x)
    maxx = np.max(x)
    xts = np.linspace(minx, maxx, 3)
    xtlabels = xts/1000
    axs[-1,j].set_xticks(ticks=xts, labels=xtlabels)
    axs[-1,j].set_xlabel('x (km)')
  for i in range(nrow):
    z = ncd.variables["z"][:]
    minz = np.min(z)
    maxz = np.max(z)
    zts = np.linspace(minz, maxz, 5)
    ztlabels = zts/1000
    axs[i, 0].set_yticks(ticks=zts, labels=ztlabels)
    axs[i, 0].set_ylabel("z (km)")

  titlestr = "supercell_example t = {0:.2f} (hr)".format(ncd.variables["t"][t_idx]/3600)
  fig.suptitle(titlestr)

  fig.savefig(args.output_file, bbox_inches="tight")
  plt.close(fig)

