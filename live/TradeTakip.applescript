set base to "/Users/m2/Desktop/trade-arastirma"
set choice to button returned of (display dialog "Trade Takip" & return & return & "Pano her gece 01:15'te kendiliğinden güncellenir." buttons {"Raporu Aç (PDF)", "Şimdi Güncelle", "Panoyu Aç"} default button "Panoyu Aç" with title "Trade Takip")
if choice is "Panoyu Aç" then
	do shell script "open " & quoted form of (base & "/rapor/takip.html")
else if choice is "Raporu Aç (PDF)" then
	do shell script "open " & quoted form of (base & "/rapor/Strateji_Arastirma_Raporu.pdf")
else
	display notification "Veriler indiriliyor, birkaç dakika sürebilir. Bitince haber vereceğim." with title "Trade Takip"
	do shell script "cd " & quoted form of base & " && (.venv/bin/python live/run_live.py >> logs/live.log 2>&1; open rapor/takip.html) > /dev/null 2>&1 &"
end if
