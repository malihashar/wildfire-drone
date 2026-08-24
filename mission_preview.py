waypoints = [
    (43.7000, -79.4000, 30.0),
    (43.7005, -79.4000, 30.0),
    (43.7005, -79.3995, 30.0),
    (43.7000, -79.3995, 30.0),
    (43.7000, -79.4000, 30.0),
]


def main():
    print("Five-point mission:")
    print()

    for i, (lat, lon, altitude) in enumerate(waypoints, 1):
        print(
            f"Point {i}: "
            f"lat={lat}, lon={lon}, altitude={altitude}m"
        )


if __name__ == "__main__":
    main()
