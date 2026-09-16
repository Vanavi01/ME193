import asyncio
from bleak import BleakScanner

async def main():
    print("Scanning for 8 seconds...")
    devices = await BleakScanner.discover(timeout=8.0)
    lego = [d for d in devices if d.name and "Motor" in d.name]
    if not lego:
        print("No LEGO motors found. Press the button on yours to wake it.")
    for d in lego:
        print(f"{d.name}   {d.address}")

asyncio.run(main())