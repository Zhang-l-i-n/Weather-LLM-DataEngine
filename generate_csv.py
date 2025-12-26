import argparse
import csv
import os
import re
import sys
import numpy as np
import pandas as pd
from data_util.get_era5_data import get_era5_data_by_vars
from datetime import datetime, timedelta
import pytz
from metpy.calc import relative_humidity_from_dewpoint
from metpy.units import units

def utc_to_beijing(utc_time):
    if isinstance(utc_time, str):
        utc_time = datetime.fromisoformat(utc_time.replace('Z', '+00:00'))
    
    utc_time = utc_time.replace(tzinfo=pytz.UTC)
    beijing_tz = pytz.timezone('Asia/Shanghai')
    beijing_time = utc_time.astimezone(beijing_tz)
    
    return beijing_time


def beijing_to_utc(beijing_time):
    if isinstance(beijing_time, str):
        beijing_time = datetime.fromisoformat(beijing_time)
    
    beijing_tz = pytz.timezone('Asia/Shanghai')
    beijing_time = beijing_tz.localize(beijing_time) if beijing_time.tzinfo is None else beijing_time
    utc_time = beijing_time.astimezone(pytz.UTC)
    return utc_time


def generate_three_hour_intervals(start_time_str, format_str="%Y-%m-%dT%H:%M:%S"):
    """
    生成从给定时间开始到第二天20:00:00的每三个小时的时间段。

    参数:
    start_time_str (str): 起始时间的字符串，例如 "2023-10-26T10:00:00"。
    format_str (str): 输入时间字符串的格式。

    返回:
    list: 包含每个时间段起始和结束时间的元组列表。
    """
    try:
        current_time = datetime.strptime(start_time_str, format_str)
    except ValueError as e:
        print(f"错误：无效的时间字符串或格式。请检查输入。错误信息：{e}")
        return []

    end_date = (current_time + timedelta(days=1)).date()
    end_time_boundary = datetime.combine(end_date, datetime.min.time()) + timedelta(hours=20)

    time_intervals = []
    
    while current_time < end_time_boundary:
        end_time = current_time + timedelta(hours=3)
        if end_time > end_time_boundary:
            end_time = end_time_boundary
        
        time_intervals.append((current_time.strftime(format_str), end_time.strftime(format_str)))
        current_time = end_time

    return time_intervals

def get_temperature_by_3h(grib_file, time_str, lat_range, lon_range, format_str = "%Y-%m-%dT%H:%M:%S"):
    """
    计算气温
    05、11时取今天最高，明天最低；17、21时取明天最高、明天最低
    """
    temperature_ds = get_era5_data_by_vars(grib_file, "t2m")

    if temperature_ds is None:
        print("无法获取温度数据，请检查文件路径或变量名。")
        return [], []

    if temperature_ds['time'].dtype == 'O':
        temperature_ds['time'] = temperature_ds['time'].astype('datetime64[ns]')

    max_temps_c = []
    min_temps_c = []

    time_intervals = generate_three_hour_intervals(time_str, format_str=format_str)
    for start_time_bjt, end_time_bjt in time_intervals:
        start_utc_str = beijing_to_utc(start_time_bjt).strftime(format_str)
        end_utc_str = beijing_to_utc(end_time_bjt).strftime(format_str)

        regional_temp = temperature_ds.sel(time=slice(start_utc_str, end_utc_str))
        regional_temp = regional_temp.sel(latitude=lat_range, longitude=lon_range)

        if regional_temp['t2m'].size > 0:
            max_temp_k = regional_temp['t2m'].max().item()
            min_temp_k = regional_temp['t2m'].min().item()

            max_temp_c = max_temp_k - 273.15
            min_temp_c = min_temp_k - 273.15

            max_temps_c.append(max_temp_c)
            min_temps_c.append(min_temp_c)
        else:
            max_temps_c.append(None)
            min_temps_c.append(None)

    return max_temps_c, min_temps_c


def get_rh_by_days(grib_file, time_str, target_lat, target_lon, format_str = "%Y-%m-%dT%H:%M:%S"):
    """
    计算相对湿度
    提取徐家汇（lon=121.4317，lat=31.1922）的露点温度DPT_2M
    根据露点温度和最高、最低气温计算最低rhmin和最高相对湿度rhmax（用metpy中的relative_humidity_from_dewpoint函数）
    """
    time_intervals = generate_three_hour_intervals(time_str, format_str=format_str)

    start_time_utc = beijing_to_utc(time_intervals[0][0]).replace(tzinfo=None)
    end_time_utc = beijing_to_utc(time_intervals[-1][-1]).replace(tzinfo=None)

    temperature_ds = get_era5_data_by_vars(grib_file, "t2m")
    dewpoint_ds = get_era5_data_by_vars(grib_file, "d2m")
    if temperature_ds is None or dewpoint_ds is None:
        print("无法获取温度或露点数据，请检查文件路径或变量名。")
        return None, None

    t2m_da = temperature_ds["t2m"]
    d2m_da = dewpoint_ds["d2m"]

    t2m_slice = t2m_da.sel(time=slice(start_time_utc, end_time_utc))
    d2m_slice = d2m_da.sel(time=slice(start_time_utc, end_time_utc))

    t2m_point = t2m_slice.sel(longitude=target_lon, latitude=target_lat, method="nearest")
    d2m_point = d2m_slice.sel(longitude=target_lon, latitude=target_lat, method="nearest")

    t2m_c = t2m_point - 273.15
    d2m_c = d2m_point - 273.15

    rh = relative_humidity_from_dewpoint(
        t2m_c.values * units.degC,
        d2m_c.values * units.degC
    )

    rhmax = np.max(rh).magnitude * 100
    rhmin = np.min(rh).magnitude * 100

    return rhmin, rhmax


def get_wdir_by_3h(grib_file, time_str, lat_range, lon_range, format_str = "%Y-%m-%dT%H:%M:%S"):
    """
    计算指定时间段内的平均风向。

    首先，函数会根据传入的 `time_str` 和 `format_str` 确定一个24小时的时间段，
    即从 `time_str` 当天开始，到第二天20:00（北京时间）为止。然后，它会
    从 GRIB 文件中提取该时间段内所有时刻的 u10 和 v10 数据，并计算它们的
    时间平均值，以得到每个格点上的平均风矢量 (u, v)。

    最后，根据风向的定义，使用以下公式计算平均风向 (wdir)：
    `wdir = 180 + arctan2(u, v) * (180 / pi)`

    该公式基于气象学中风向的定义，即风吹来的方向，角度从正北（0度）开始，
    顺时针增加。`arctan2(u, v)` 计算的是风吹向的方向，因此需要进行180度
    的调整以得到风吹来的方向。

    参数:
    - grib_file (str): GRIB 文件的路径。
    - time_str (str): 包含日期的字符串，格式由 `format_str` 指定。
    - format_str (str): 用于解析 `time_str` 的时间格式。

    返回:
    - numpy.ndarray: 包含每个格点平均风向的二维数组。
    """

    intervals = generate_three_hour_intervals(time_str, format_str=format_str)

    wdir_list = []
    u10_ds = get_era5_data_by_vars(grib_file, "u10")
    v10_ds = get_era5_data_by_vars(grib_file, "v10")
    if u10_ds is None or v10_ds is None:
        print("无法获取风向数据，请检查文件路径或变量名。")
        return None

    u10_da = u10_ds["u10"]
    v10_da = v10_ds["v10"]

    for interval_start, interval_end in intervals:
        start_utc = beijing_to_utc(interval_start).replace(tzinfo=None)
        end_utc = beijing_to_utc(interval_end).replace(tzinfo=None)
        u_slice = u10_da.sel(time=slice(start_utc, end_utc)).sel(latitude=lat_range, longitude=lon_range)
        v_slice = v10_da.sel(time=slice(start_utc, end_utc)).sel(latitude=lat_range, longitude=lon_range)
        # 时间均值
        u = u_slice.mean(dim='time')
        v = v_slice.mean(dim='time')
        deg = 180.0 / np.pi
        wdir = 180.0 + np.arctan2(u, v) * deg
        # 所有经纬度均值
        wdir_mean = wdir.mean().item()
        wdir_list.append(wdir_mean)

    return wdir_list


def get_uvg_by_3h(grib_file, time_str, lat_range, lon_range, format_str = "%Y-%m-%dT%H:%M:%S"):
    """
    计算风速，用inner的计算方法
    小时最大阵风转化为等级
    """
    def ws2scale_city(ws):
        undef=32767
        if np.abs(ws)!=undef:
            if 0<=ws<5.0:
                scale=2
            elif 5.0<=ws<8.0:
                scale=4
            elif 8.0<=ws<10:
                scale=5
            elif 10<=ws<12:
                scale=6
            elif 12<=ws<14:
                scale=6.5
            elif 14<=ws<16:
                scale=7
            elif 16<=ws<18:
                scale=7.5
            elif 18<=ws<20:
                scale=8
            elif 20<=ws<22:
                scale=8.5
            elif 22<=ws<26:
                scale=9
            elif 26<=ws<27.5:
                scale=9.5
            elif 27.5<=ws<28.5:
                scale=10
            elif 28.5<=ws<30:
                scale=10.5
            elif 29.5<=ws<31.5:
                scale=11
            elif 31.5<=ws<33.5:
                scale=11.5
            elif 33.5<=ws<35.5:
                scale=12
            elif 35.5<=ws<37.5:
                scale=12.5
            elif 37.5<=ws<39.5:
                scale=13
            elif 39.5<=ws<41.5:
                scale=13.5
            elif 41.5<=ws<43.9:
                scale=14
            elif 43.9<=ws<46.2:
                scale=14.5
            elif 46.2<=ws<48.7:
                scale=15
            elif 48.7<=ws<51.0:
                scale=15.5
            elif 51.0<=ws<53.6:
                scale=16
            elif 53.6<=ws<56.1:
                scale=16.5
            elif 56.1<=ws<58.7:
                scale=17
            elif 58.7<=ws<61.3:
                scale=17.5
            else:
                scale=np.nan
        return scale

    wndg = get_era5_data_by_vars(grib_file, "i10fg")
    if wndg is None:
        print("无法获取风速数据，请检查文件路径或变量名。")
        return None
    
    wndg_da = wndg["i10fg"]
    
    intervals = generate_three_hour_intervals(time_str, format_str=format_str)
    uvg_list = []
    for interval_start, interval_end in intervals:
        start_utc = beijing_to_utc(interval_start).replace(tzinfo=None)
        end_utc = beijing_to_utc(interval_end).replace(tzinfo=None)
        wndg_slice = wndg_da.sel(time=slice(start_utc, end_utc)).sel(latitude=lat_range, longitude=lon_range)
        if wndg_slice.size == 0:
            uvg_list.append(None)
            continue
        max_wind_speed = wndg_slice.max().item()
        uvg_inner = ws2scale_city(max_wind_speed)
        uvg_list.append(uvg_inner)
    
    return uvg_list


def get_cloud_by_3h(land_file, level_file, time_str, lat_range, lon_range, format_str = "%Y-%m-%dT%H:%M:%S"):
    """
    计算云量代码
    需要tcc和lcc（总云量、低云量），rh850和rh700（850/700hPa相对湿度），tp（降水）
    返回云量代码cloud: 0, 1, 2
    """

    tcc_ds = get_era5_data_by_vars(land_file, "tcc")
    lcc_ds = get_era5_data_by_vars(land_file, "lcc")
    tp_ds = get_era5_data_by_vars(land_file, "tp")
    rh_ds = get_era5_data_by_vars(level_file, "r")
    
    if tcc_ds is None or lcc_ds is None or rh_ds is None or tp_ds is None:
            print("无法获取云量或湿度数据，请检查文件路径或变量名。")
            return None
    
    cloud_list = []
    intervals = generate_three_hour_intervals(time_str, format_str=format_str)
    for start_time_bjt, end_time_bjt in intervals:
        start_time_utc = beijing_to_utc(start_time_bjt).replace(tzinfo=None)
        end_time_utc = beijing_to_utc(end_time_bjt).replace(tzinfo=None)
        tcc = tcc_ds["tcc"].sel(time=slice(start_time_utc, end_time_utc)).sel(latitude=lat_range, longitude=lon_range)
        lcc = lcc_ds["lcc"].sel(time=slice(start_time_utc, end_time_utc)).sel(latitude=lat_range, longitude=lon_range)
        tp = tp_ds["tp"].sel(time=slice(start_time_utc, end_time_utc)).sel(latitude=lat_range, longitude=lon_range)
        if "isobaricInhPa" in rh_ds["r"].dims:
            rh850 = rh_ds["r"].sel(time=slice(start_time_utc, end_time_utc), isobaricInhPa=850).sel(latitude=lat_range, longitude=lon_range)
            rh700 = rh_ds["r"].sel(time=slice(start_time_utc, end_time_utc), isobaricInhPa=700).sel(latitude=lat_range, longitude=lon_range)
        else:
            print("湿度数据缺少isobaricInhPa维度")
            return None
        
        rh850 = rh850 * 100
        rh700 = rh700 * 100
        tcc = tcc * 100
        lcc = lcc * 100
        # 条件一
        cond1 = (rh850 + rh700 < 140) & (rh850 < 80) & (rh700 < 80)
        tcc = tcc.where(~cond1, tcc * ((rh700 + rh850) / 200) ** 0.4)
        # 条件二
        cond2 = ((rh850 < 40) & (rh700 < 30)) | (rh700 < 10)
        lcc = lcc.where(~cond2, lcc * (rh850 / 100) ** 0.3)
        # 条件三
        tp_aligned = tp.reindex_like(lcc)
        cond3 = (tp_aligned < 0.3)
        lcc = lcc.where(~cond3, lcc / 2)

        meanlcc = np.nanmean(lcc)
        meantcc = np.nanmean(tcc)

        if ((meantcc > 80) and (meanlcc > 40)) or (meantcc > 90) or (meanlcc > 90):
            cloud = 2
        elif (meanlcc > 15) or (meantcc > 40):
            cloud = 1
        else:
            cloud = 0
        
        cloud_list.append(cloud)

    return cloud_list


def get_rain_by_3h(grib_file, time_str, lat_range, lon_range, format_str = "%Y-%m-%dT%H:%M:%S"):
    start_time_bjt = datetime.strptime(time_str, format_str)
    end_time_bjt = start_time_bjt + timedelta(days=1)
    end_time_bjt = end_time_bjt.replace(hour=20, minute=0, second=0, microsecond=0)

    intervals = []
    t = start_time_bjt
    while t < end_time_bjt:
        interval_start = t
        interval_end = t + timedelta(hours=3)
        if interval_end > end_time_bjt:
            interval_end = end_time_bjt
        intervals.append((interval_start, interval_end))
        t = interval_end

    tp_ds = get_era5_data_by_vars(grib_file, "tp")
    sf_ds = get_era5_data_by_vars(grib_file, "sf")
    cp_ds = get_era5_data_by_vars(grib_file, "cp")
    if tp_ds is None or sf_ds is None or cp_ds is None:
        print("无法获取降水相关数据，请检查文件路径或变量名。")
        return None, None, None

    tp_da = tp_ds["tp"] * 1000  # 转化为mm
    sf_da = sf_ds["sf"] * 1000
    cp_da = cp_ds["cp"] * 1000

    ifrain_list, tpmax_list, rain_percent_list = [], [], []
    for interval_start, interval_end in intervals:
        start_utc = beijing_to_utc(interval_start).replace(tzinfo=None)
        end_utc = beijing_to_utc(interval_end).replace(tzinfo=None)

        tp = tp_da.sel(time=slice(start_utc, end_utc)).sel(latitude=lat_range, longitude=lon_range)
        sf = sf_da.sel(time=slice(start_utc, end_utc)).sel(latitude=lat_range, longitude=lon_range)
        
        cp = cp_da.sel(time=slice(start_utc, end_utc)).sel(latitude=lat_range, longitude=lon_range)
        tp = tp.reindex_like(sf)
        cp = cp.reindex_like(sf)

        mon = interval_start.month
        if mon in range(4, 11):
            tp = tp.where(tp >= 0.2)
        else:
            tp = tp.where(tp >= 0.15)

        sleet = tp.where(((tp - sf) > 0.1) & (sf > 0.1))
        snow = sf.where(((tp - sf) < 0.1) & (sf > 0.1))
        print(np.sum(snow).item())
        rain = tp.where(((tp - sf) > 0.1) & (sf < 0.1))
        thunder = tp.where((~np.isnan(rain)) & (cp >= 5))
        drizzle = tp.where((~np.isnan(rain)) & (cp > 2) & (cp < 5))

        count_precip = np.sum((~np.isnan(rain)) | (~np.isnan(sleet)) | (~np.isnan(snow))).item()
        count_total = tp.size # 或者使用 mask.size
        count_sleet = np.sum(~np.isnan(sleet)).item()
        count_snow = np.sum(~np.isnan(snow)).item()
        count_rain = np.sum(~np.isnan(rain)).item()
        count_thunder = np.sum(~np.isnan(thunder)).item()
        count_drizzle = np.sum(~np.isnan(drizzle)).item()

        if tp.size == 0 or np.all(np.isnan(tp)):
            tpmax = 0
        else:
            tpmax = np.nanmax(tp).item()
        tpmax_list.append(tpmax)

        rain_percent = count_precip / count_total if count_total > 0 else 0
        rain_percent_list.append(rain_percent)
        rain_type = [0, 0, 0, 0, 0]
        if count_total > 0:
            if count_sleet / count_total >= 0.05:
                rain_type[0] = 1
            if count_snow / count_total >= 0.05:
                rain_type[1] = 1
            if count_rain / count_total >= 0.05:
                rain_type[2] = 1
            if count_thunder / count_total >= 0.05:
                rain_type[3] = 1
            if count_drizzle / count_total >= 0.05:
                rain_type[4] = 1

        ifrain = 0
        if count_precip == 0:
            ifrain_list.append(ifrain)
            continue
        # 夏半年 (4-10月)
        if mon in range(4, 11):
            if rain_type[3] == 1:  # 有雷阵雨
                if rain_percent >= 0.8:
                    ifrain = 3.0
                elif 0.5 <= rain_percent < 0.8:
                    ifrain = 3.1
                else: # 0 < rain_percent < 0.5
                    ifrain = 3.2
            elif rain_type[4] == 1: # 有阵雨 (非雷阵雨)
                if rain_percent >= 0.8:
                    ifrain = 2.0
                elif 0.5 <= rain_percent < 0.8:
                    ifrain = 2.1
                else: # 0 < rain_percent < 0.5
                    ifrain = 2.2
            elif rain_type[2] == 1: # 有雨 (非阵雨或雷阵雨)
                if rain_percent >= 0.8:
                    if np.nanmax(rain.data) >= 5:
                        ifrain = 1.0
                    else:
                        ifrain = 1.1
                elif 0.5 <= rain_percent < 0.8:
                    ifrain = 1.2
                elif 0.2 <= rain_percent < 0.5:
                    ifrain = 1.3
                else: # 0 < rain_percent < 0.2
                    ifrain = 1.4
            else:
                ifrain = 0 # 没有满足5%阈值的降水类型

        # 冬半年 (11-3月)
        else:
            if rain_type[1] == 1: # 有雪
                # 有阵雨和/或雨夹雪/雨
                if rain_type[4] == 1 or rain_type[0] == 1 or rain_type[2] == 1: 
                    # 有雪或雨夹雪/雨
                    if count_sleet + count_rain >= count_snow: 
                        if rain_percent >= 0.8:
                            ifrain = 11.0 # 阵雨夹雪或雪
                        elif 0.5 <= rain_percent < 0.8:
                            ifrain = 11.1
                        else:
                            ifrain = 11.2
                    # 雪或雨夹雪/雨
                    else:
                        if rain_percent >= 0.8:
                            ifrain = 15.0 # 阵雪或阵雨夹雪
                        elif 0.5 <= rain_percent < 0.8:
                            ifrain = 15.1
                        else:
                            ifrain = 15.2
                else: # 只有雪，没有阵雨、雨夹雪、雨
                    if rain_percent >= 0.8:
                        if np.nanmax(snow.data) >= 0.5:
                            ifrain = 12.0
                        else:
                            ifrain = 12.1
                    elif 0.5 <= rain_percent < 0.8:
                        ifrain = 12.2
                    else:
                        ifrain = 12.4
            
            # 没有雪，但有其他降水
            elif rain_type[0] == 1 or rain_type[2] == 1: # 有雨夹雪或雨
                # 有阵雨
                if rain_type[4] == 1:
                    # 有雨和/或雨夹雪
                    if rain_type[2] == 1 and rain_type[0] == 1:
                        if count_rain >= count_sleet:
                            if rain_percent >= 0.8: ifrain = 5.0 # 阵雨或阵雨夹雪
                            elif 0.5 <= rain_percent < 0.8: ifrain = 5.1
                            else: ifrain = 5.2
                        else:
                            if rain_percent >= 0.8: ifrain = 9.0 # 阵雨夹雪或阵雨
                            elif 0.5 <= rain_percent < 0.8: ifrain = 9.1
                            else: ifrain = 9.2
                    elif rain_type[0] == 1: # 只有阵雨夹雪
                        if rain_percent >= 0.8: ifrain = 7.0
                        elif 0.5 <= rain_percent < 0.8: ifrain = 7.1
                        else: ifrain = 7.2
                    elif rain_type[2] == 1: # 只有阵雨
                        if rain_percent >= 0.8: ifrain = 2.0
                        elif 0.5 <= rain_percent < 0.8: ifrain = 2.1
                        else: ifrain = 2.2
                # 没有阵雨
                else:
                    if rain_type[2] == 1 and rain_type[0] == 1:
                        if count_rain >= count_sleet:
                            if rain_percent >= 0.8: 
                                if np.nanmax(rain.data) >= 5: ifrain = 4.0
                                else: ifrain = 4.1
                            elif 0.5 <= rain_percent < 0.8: ifrain = 4.2
                            else: ifrain = 4.4
                        else:
                            if rain_percent >= 0.8: 
                                if np.nanmax(rain.data) >= 5: ifrain = 8.0
                                else: ifrain = 8.1
                            elif 0.5 <= rain_percent < 0.8: ifrain = 8.2
                            else: ifrain = 8.4
                    elif rain_type[0] == 1: # 只有雨夹雪
                        if rain_percent >= 0.8: 
                            if np.nanmax(sleet.data) >= 3: ifrain = 6.0
                            else: ifrain = 6.1
                        elif 0.5 <= rain_percent < 0.8: ifrain = 6.2
                        else: ifrain = 6.4
                    elif rain_type[2] == 1: # 只有雨
                        if rain_percent >= 0.8: 
                            if np.nanmax(rain.data) >= 5: ifrain = 1.0
                            else: ifrain = 1.1
                        elif 0.5 <= rain_percent < 0.8: ifrain = 1.2
                        else: ifrain = 1.4

        ifrain_list.append(ifrain)

    return ifrain_list, tpmax_list, rain_percent_list


def parse_args():
    parser = argparse.ArgumentParser(description="处理 ERA5 GRIB 数据并生成天气预报 CSV")
    
    parser.add_argument("--land_file", type=str, default='./raw_data/land.grib', help="地面层 GRIB 文件路径")
    parser.add_argument("--level_file", type=str, default='./raw_data/level.grib', help="高空层 GRIB 文件路径")
    parser.add_argument("--output_dir", type=str, default='./forecast_csv', help="输出 CSV 的目录")
    
    parser.add_argument("--start_date", type=str, default="2021-01-01", help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2021-01-01", help="结束日期 (YYYY-MM-DD)")
    
    return parser.parse_args()

if __name__ == '__main__': 
    args = parse_args()
    
    # 1. 检查文件是否存在
    if not os.path.exists(args.land_file) or not os.path.exists(args.level_file):
        print(f"❌ 错误: 输入数据文件不存在。请检查路径:\n  - {args.land_file}\n  - {args.level_file}")
        sys.exit(1)
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    hours_to_run = [5, 11, 17, 20]
    
    current_date = start_date
    print(f"--- 开始处理数据: {args.start_date} 至 {args.end_date} ---")
    
    while current_date <= end_date:
        for hour in hours_to_run:
            start_time = datetime(current_date.year, current_date.month, current_date.day, hour, 0, 0).strftime("%Y-%m-%dT%H:%M:%S")

            print(f"正在处理: {start_time} ...", end="", flush=True)

            time_intervals_bjt = [i[0] for i in generate_three_hour_intervals(start_time)]

            DEFAULT_LON_SLICE = 121.4317
            DEFAULT_LAT_SLICE = 31.1922
            DEFAULT_TARGET_LAT = slice(32, 30.5)
            DEFAULT_TARGET_LON = slice(120.5, 122)
            
            # 提取数据
            max_temps_c, min_temps_c = get_temperature_by_3h(args.land_file, start_time, DEFAULT_LAT_SLICE, DEFAULT_LON_SLICE)
            rh_min, rh_max = get_rh_by_days(args.land_file, start_time, DEFAULT_TARGET_LAT, DEFAULT_TARGET_LON)
            wdir_list = get_wdir_by_3h(args.land_file, start_time, DEFAULT_LAT_SLICE, DEFAULT_LON_SLICE)
            uvg_list = get_uvg_by_3h(args.land_file, start_time, DEFAULT_LAT_SLICE, DEFAULT_LON_SLICE)
            cloud_list = get_cloud_by_3h(args.land_file, args.level_file, start_time, DEFAULT_LAT_SLICE, DEFAULT_LON_SLICE)
            ifrain_list, tpmax_list, rain_percent_list = get_rain_by_3h(args.land_file, start_time, DEFAULT_LAT_SLICE, DEFAULT_LON_SLICE)

            if not ifrain_list:
                print(" [失败] 数据提取不完整")
                continue

            data = pd.DataFrame({
                'fsttime': time_intervals_bjt,
                'max_temp_c': max_temps_c,
                'min_temp_c': min_temps_c,
                'rhmin': [rh_min] * len(time_intervals_bjt),
                'rhmax': [rh_max] * len(time_intervals_bjt),
                'wdir': wdir_list,
                'uvg': uvg_list,
                'cloud': cloud_list,
                'ifrain': ifrain_list,
                'tpmax': tpmax_list,
                'rain_percent': rain_percent_list
            })


            file_name = f"{start_time.replace(':', '').replace('T', '_')}.csv"
            save_path = os.path.join(args.output_dir, file_name)
            data.to_csv(save_path, index=False)
            print(f" -> {file_name}")

        current_date += timedelta(days=1)
    
    print("--- 处理完成 ---")