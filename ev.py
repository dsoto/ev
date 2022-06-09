# library of convenience functions for analyzing EV data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display

# function returns section of data germane to acceleration analysis
# should pull section of data with motor current above a certain threshold
def pull_throttle_region(ride_data):
    threshold = 0
    # find areas of motor throttle above threshold current
    above_threshold = (ride_data['current_motor'] > 0).astype(int)
    # data between +1 and -1 is contiguous region above threshold
    diffs = above_threshold.diff()
    transitions = diffs[diffs != 0]
    # get index of first +1
    one_locations = np.where(transitions==1)[0]
    first_one = one_locations[0]
    start = transitions.index[first_one]
    # get index of first -1 after first +1
    neg_one_locations = np.where(transitions==-1)[0]
    first_neg_one_after_first_one = neg_one_locations[np.where(neg_one_locations > first_one)[0][0]]
    end = transitions.index[first_neg_one_after_first_one]
    # return data sliced by time
    return ride_data.loc[start:end]

def reset_time_index(ride_data):
    ride_data.index = ride_data.index - ride_data.index[0]
    return ride_data

def create_speed_gps_erpm(ride_data):
    # normalize erpm and then multiply by max GPS speed
    return ride_data['erpm'] / ride_data['erpm'].max() * ride_data['gnss_gVel'].max()

def calculate_acceleration(data, window_length=None):
    def calc_slope(data):
        return np.polyfit(data.index.values, data.values, 1)[0]

    if window_length == None:
        window_length = 35
    return data['speed'].rolling(window_length, center=True).apply(calc_slope)

def rpm_to_radpsec(data, pole_pairs=None):
    return data['erpm'] / pole_pairs / 60 * 2 * 3.1415

def plot_torque_speed(data):
    fig, ax = plt.subplots()
    sns.scatterplot(data=data,
                    y='current_motor',
                    x='rad_per_sec',
                    ax=ax)
    ax.set_xlabel('Motor Speed (rad/sec)')
    ax.set_ylabel('Motor Current (A)')
    ax.set_title('Observed Torque-Speed Curve')
    ax.grid(True)

def plot(ride_data):
    columns = ['battery_current',
               'motor_current',
               'battery_temperature',
               'motor_temperature',
               'controller_temperature']
    ride_data[columns].plot()

def vesc_speed(vesc_data):
    wheel_diameter_m = 0.567
    pole_pairs = 23
    rpm_to_mph = 60 * wheel_diameter_m * 3.14 / 1609 /pole_pairs    # rev/min * min/hour * meter/rev * mile/meter
    rpm_to_mps = 1 / 60 * wheel_diameter_m * 3.14 / pole_pairs       # rev/min * min/sec * meter/rev
    vesc_data['speed_mps'] = vesc_data['erpm'] * rpm_to_mps
    return vesc_data

def ride_report(filename):
    data = vesc_csv_to_df(filename)
    data = pull_throttle_region(data)
    data = reset_time_index(data)
    data = vesc_convert_to_seconds(data)
    data['speed'] = create_speed_gps_erpm(data)
    data['acceleration'] = calculate_acceleration(data)
    data['rad_per_sec'] = rpm_to_radpsec(data, pole_pairs=23)

    plot_vesc(data)
    plot_current_acceleration(data)
    plot_internal_resistance(data)
    plot_torque_speed(data)

def output_maximums(data):
    print(f'max phase current {data.current_motor.max():.1f} A')
    print(f'max battery current {data.current_in.max():.1f} A')
    print(f'max acceleration {data.acceleration.max():.2f} m/sec^2')
    print(f'max speed {data.speed.max():.2f} m/sec^2')

    markdown = f'max phase current {data.current_motor.max():.1f} A\n'
    markdown += f'max battery current {data.current_in.max():.1f} A\n'
    markdown += f'max acceleration {data.acceleration.max():.2f} m/sec^2\n'
    markdown += f'max speed {data.speed.max():.2f} m/sec^2'

    return markdown
    #IPython.display.Markdown(markdown)

def plot_vesc(vesc_data, ax=None):
    columns = ['current_motor',
               'current_in',
               'input_voltage',
               #'speed_mps',
               #'temp_motor'
               #'duty_cycle'
               ]
    vesc_data[columns].plot(ax=ax)

def plot_current_acceleration(data):
    fig, ax = plt.subplots()
    color = 'tab:orange'
    ax.plot(data.index, data['current_motor'], color=color)
    ax.set_ylabel('Motor Current (A)', color=color)
    ax.tick_params(axis='y', labelcolor=color)
    axa = ax.twinx()
    color = 'tab:blue'
    axa.plot(data.index, data['acceleration'], color=color)
    axa.set_ylabel('Acceleration (m/sec$^2$)', color=color)
    axa.tick_params(axis='y', labelcolor=color)
    ax.set_xlabel('Time (sec)')

def plot_internal_resistance(data):
    fit = np.polyfit(data['current_in'], data['input_voltage'], 1)
    x = np.linspace(data['current_in'].min(), data['current_in'].max(), 10)
    y = np.polyval(fit, x)
    fig, ax = plt.subplots()
    ax.plot(data['current_in'], data['input_voltage'], 'k.', alpha=0.2)
    ax.plot(x, y)
    ax.text(0.5, 0.7, f'IR (m$\Omega$): {-fit[0]*1000:.1f}',
            transform=ax.transAxes,
            fontsize=14)

    ax.set_xlabel('Battery Current (A)')
    ax.set_ylabel('Battery Voltage (V)')
    fig.suptitle('Voltage Sag and Internal Resistance')

def plot_performance(ride_data):
    columns = ['motor_current', 'battery_voltage', 'battery_current', 'speed_mph', 'power_electrical']
    ride_data[columns].plot()

def vesc_csv_to_df(filename):
    return pd.read_csv(filename, sep=';', index_col=0)

def vesc_trim_threshold(vd, column, value):
    start = vd[vd[column]>value].index[0]
    vd.index = vd.index - start
    return vd

def vesc_convert_to_seconds(vesc_data):
    vesc_data.index = vesc_data.index / 1000
    vesc_data.index.names = ['time_sec']
    return vesc_data

def motor_temp_delta(vesc_data):
    return vesc_data.iloc[-1]['temp_motor'] - vesc_data.iloc[0]['temp_motor']

def controller_temp_delta(vesc_data):
    return vesc_data.iloc[-1]['temp_mos_max'] - vesc_data.iloc[0]['temp_mos_max']