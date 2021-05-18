THERMISTOR: PARTS

Thermistor beads
Source: digikey, 223-1558-ND
Manufacturer: TE Connectivity Measurement Specialties, 	GAG22K7MCD419

Thermistor pins
Source: digikey, ED1110-ND
Manufacturer: Mill-Max Manufacturing Corp., 861-13-050-10-002000
Note: As of earlier this year, these pins no longer mate well together. See alternative (but larger) option below. 

Alternative thermistor pins 
Source: digikey, 	455-3170-ND
Manufacturer: JST Sales America Inc., A02KR02DS28W51B


THERMISTOR: BUILD

1.	Clip thermistor excess wire to a couple cm from the glass bead. If you plan to reuse thermistors, it is best to clip extra wire as you will likely need to re-solder pins following a recovery. 
2.	Separate and strip individual wires at tail end using a razor blade. 
3.	Solder two wires into the holes of the through-hole gold pins or to the leads of the alternative pins (you’ll need to remove existing wire first). 
a.	Gold through-hole pins: these come in a strip so you’ll need to clip them to get a block of two at a time. 
b.	Alternative pins: these come already attached…you’ll need to remove the existing wires and pair down part to as small as you can get it. I use clippers and a Dremel to reduce size and I solder directly to the metal leads. 
4.	Depending on delicacy, apply super glue to pin connections to strengthen and protect the connection. 


THERMISTOR: SURGERY

1.	Expose and clean skull at nasal bones. Make sure you clear enough space for your pins as well. 
2.	Drill hole through nasal bone on one side leaving underlying epithelium intact. Slightly anterior to where the nasal bones join (where the “M” meets) tends to be the best location. 
3.	Place thermistor bead so end of bead is fully below the nasal bone (the rest can protrude) without pressing on the underlying epithelium and glue in place. 
a.	I like to use a viscous super glue so it does not seep into the nasal cavity. 
b.	The less bleeding/swelling the better. In an ideal implant, you will see the nasal epithelium moving with breath. 
4.	Pins can be glued wherever convenient for your implant as long as they are accessible to plug in later. Ensure that you encase entire thermistor wire with glue for protection. 

THERMISTOR: RECOVERY

Thermistors can be recovered from expired mouse by soaking entire implant in acetone until glue is removed. 

COMMUTATOR/WIRE/AMPLIFICATION: PARTS

These are the parts we use for recording thermistor sniff signals from freely moving mice. 

Commutator
Source: adafruit, 736

Connecting wire
Source: amazon, B008AGUDEY
Manufacturer: XINYIYUAN, P/N B-30-1000 30AWG
Note: It is important to have a stiff enough wire to work with the commutator, but thin enough that it does not interfere with any camera tracking. We have found this wire works quite well. Link below. 
https://www.amazon.com/B-30-1000-30AWG-Plated-Copper-Wrepping/dp/B008AGUDEY/ref=sr_1_1?keywords=P%2FN+B-30-1000+30AWG+Tin+Plated+Copper+Wire+Wrepping+Cable+Reel+Black&qid=1561142706&s=gateway&sr=8-1

Op-amp (used in amplification circuit below) 
Source: adafruit, 808

THERMISTOR: AMPLIFICATION

These thermistors are simply variable resistors and need to be run through some form of voltage divider circuit to read out a voltage. We have included a PCBExpress circuit using an adafruit op-amp and some resistors to voltage divide and amplify the thermistor signal at a readable level. This board runs on 5V power.
