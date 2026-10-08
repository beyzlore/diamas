set site to "https://beyzlore.github.io/trade-takip/"
set choice to button returned of (display dialog "Trade Takip" & return & return & "Pano her iş günü ABD kapanışından sonra bulutta kendiliğinden güncellenir (bilgisayarın kapalı olsa bile)." buttons {"Raporu Aç (PDF)", "Şimdi Güncelle", "Panoyu Aç"} default button "Panoyu Aç" with title "Trade Takip")
if choice is "Panoyu Aç" then
	open location site
else if choice is "Raporu Aç (PDF)" then
	open location site & "Strateji_Arastirma_Raporu.pdf"
else
	do shell script "/opt/homebrew/bin/gh workflow run daily.yml -R beyzlore/trade-takip"
	display notification "Bulutta güncelleme başlatıldı, yaklaşık 10 dakika sürer." with title "Trade Takip"
end if
