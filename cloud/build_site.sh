#!/usr/bin/env bash
# Web sayfasını (GitHub Pages) hazırlar.
set -e
rm -rf site && mkdir -p site
cp rapor/takip.html site/index.html
cp rapor/rapor.html site/rapor.html
cp rapor/Strateji_Arastirma_Raporu.pdf site/ 2>/dev/null || true
touch site/.nojekyll
