# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 10:14:14 2026

@author: umroot
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.graphics.tsaplots import plot_acf





#### temp variation
def temperature_visualization(airTemp):
    plt.figure(figsize=(14,5))
    plt.plot(airTemp.index, airTemp)
    # major ticks every 2 months
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    # date format
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    # rotate labels
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
def load_visualization(airTemp,load):
    fig, ax = plt.subplots(figsize=(6.5, 3.2))

    ax.plot(
        airTemp.index[:int(len(airTemp)*15/24)],#[-int(len(tgt_airTemp)*9/24):],
        load[:int(len(airTemp)*15/24)],#[-int(len(tgt_airTemp)*9/24):],
        color="black",
        linewidth=1.5
    )

    # X-axis: every 2 months
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # Axis labels
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Load (kWh)", fontsize=11)

    # Tick formatting
    ax.tick_params(axis="both", which="major", labelsize=9)

    # Rotate dates
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Subtle grid
    ax.grid(
        True,
        axis="y",
        linestyle="--",
        linewidth=0.5,
        alpha=0.4
    )

    # Remove unnecessary borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Clean layout
    fig.tight_layout()

    # Save as vector graphic for paper
    # fig.savefig(
    #     "load.pdf",
    #     bbox_inches="tight"
    # )

    plt.show()
    
def acf_load(load):
    plt.figure(figsize=(12,5))
    plot_acf(
        load,
        lags=24*14,   # 14 days for hourly data
        zero=False
    )
    plt.xlabel('Lag (hours)')
    plt.ylabel('Autocorrelation')
    plt.show()