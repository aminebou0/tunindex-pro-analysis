const API_BASE_URL = `${window.location.protocol}//${window.location.host}`; // URL dynamique pour Flask

// Variable globale pour indiquer si les données sont chargées
let isDataLoaded = false; 
let allMetricsData = null; // Pour stocker toutes les métriques récupérées

document.addEventListener('DOMContentLoaded', function() {
    // Initialiser les tooltips Bootstrap
    $(function () {
        $('[data-toggle="tooltip"]').tooltip({
            container: 'body', // Important pour éviter les problèmes de z-index
            trigger: 'hover focus' // Afficher au survol et au focus pour accessibilité
        });
    });

    // Navigation
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
    const pageSections = document.querySelectorAll('.page-section');

    function showSection(targetId) {
        pageSections.forEach(section => {
            section.style.display = (section.id === targetId) ? 'block' : 'none';
        });
        navLinks.forEach(l => {
            l.classList.toggle('active', l.getAttribute('data-section') === targetId);
        });
    }

    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('data-section');
            showSection(targetId);
            localStorage.setItem('activeSectionTUNIndex', targetId);
        });
    });

    const savedSection = localStorage.getItem('activeSectionTUNIndex');
    if (savedSection && document.getElementById(savedSection)) {
        showSection(savedSection);
    } else {
        if (pageSections.length > 0) showSection(pageSections[0].id);
    }
    
    // Dark Mode Toggle
    const darkModeToggle = document.getElementById('darkModeToggle');
    const darkModeText = document.getElementById('darkModeText');
    function setDarkMode(isDark) {
        document.body.classList.toggle('dark-mode', isDark);
        localStorage.setItem('darkModeTUNIndex', isDark);
        darkModeText.textContent = isDark ? 'Mode Clair' : 'Mode Sombre'; // Traduit
        darkModeToggle.querySelector('i').className = isDark ? 'fas fa-sun' : 'fas fa-moon';
    }
    darkModeToggle.addEventListener('click', () => {
        setDarkMode(!document.body.classList.contains('dark-mode'));
    });
    setDarkMode(localStorage.getItem('darkModeTUNIndex') === 'true');
    
    updateButtonStates(); 

    document.getElementById('filterModel').addEventListener('change', renderMetricsTableWithFilters);
    document.getElementById('filterSet').addEventListener('change', renderMetricsTableWithFilters);
    document.getElementById('filterMetricName').addEventListener('change', renderMetricsTableWithFilters);

    const fileUploadInput = document.getElementById('fileUpload');
    if (fileUploadInput) {
        fileUploadInput.addEventListener('change', function(e){
            const fileName = e.target.files[0] ? e.target.files[0].name : 'Choisir un fichier...';
            const nextSibling = e.target.nextElementSibling;
            if (nextSibling && nextSibling.classList.contains('custom-file-label')) {
                nextSibling.innerText = fileName;
            }
        });
    }

    // Gestion du bouton pliable pour les détails du fichier
    const toggleUploadInfoBtn = document.getElementById('toggleUploadInfoBtn');
    if (toggleUploadInfoBtn) {
        toggleUploadInfoBtn.addEventListener('click', function() {
            const icon = this.querySelector('i');
            // La classe 'collapsed' est gérée par Bootstrap sur le bouton lorsque la cible est masquée
            const isTargetCollapsed = !$('#uploadInfoDetails').hasClass('show');
            if (isTargetCollapsed) { // Si on va l'ouvrir
                icon.className = 'fas fa-chevron-up';
            } else { // Si on va le fermer
                icon.className = 'fas fa-chevron-down';
            }
        });
        // S'assurer que l'icône est correcte au chargement si l'élément est déjà ouvert/fermé
         $('#uploadInfoDetails').on('shown.bs.collapse', function () {
            if(toggleUploadInfoBtn) toggleUploadInfoBtn.querySelector('i').className = 'fas fa-chevron-up';
        });
        $('#uploadInfoDetails').on('hidden.bs.collapse', function () {
            if(toggleUploadInfoBtn) toggleUploadInfoBtn.querySelector('i').className = 'fas fa-chevron-down';
        });

    }

    // Afficher/Masquer les options PDF
    const exportFormatSelect = document.getElementById('exportFormat');
    const pdfOptionsDiv = document.getElementById('pdfOptions');
    if (exportFormatSelect && pdfOptionsDiv) {
        exportFormatSelect.addEventListener('change', function() {
            if (this.value === 'pdf') {
                pdfOptionsDiv.style.display = 'block';
            } else {
                pdfOptionsDiv.style.display = 'none';
            }
        });
    }
});

function updateButtonStates() {
    const buttonsToDisable = [
        ...document.querySelectorAll('#overview-section button, #analysis-section button, #modeling-section button, #prediction-section button, #export-section button')
    ];
     buttonsToDisable.forEach(button => {
        if (button.getAttribute('onclick') === 'fetchDataOverview()' && document.getElementById('overview-section').style.display === 'block') {
            // Le bouton pour charger l'aperçu reste actif
        } else if (button.getAttribute('onclick') !== 'uploadFile()') { 
             button.disabled = !isDataLoaded;
        }
    });
}


function showLoading(elementId, message = "Chargement...") { // Traduit
    const targetElement = document.getElementById(elementId);
    if (targetElement) {
        targetElement.innerHTML = `<div class="d-flex align-items-center justify-content-center p-3">
                                      <div class="spinner-border text-primary mr-2" role="status" style="width: 2rem; height: 2rem;">
                                          <span class="sr-only">Chargement...</span>
                                      </div>
                                      <strong class="text-primary">${message}</strong>
                                  </div>`;
    }
}

function showAlert(message, type = 'info', duration = 5000) {
    const alertContainer = document.getElementById('alert-container');
    // Supprimer les alertes existantes pour éviter l'accumulation
    while (alertContainer.firstChild) {
        alertContainer.removeChild(alertContainer.firstChild);
    }

    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show m-1`;
    alertDiv.role = 'alert';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="close" data-dismiss="alert" aria-label="Fermer">
            <span aria-hidden="true">&times;</span>
        </button>
    `;
    alertContainer.appendChild(alertDiv);

    if (duration !== null) {
        setTimeout(() => {
            if (window.jQuery) {
                $(alertDiv).alert('close');
            } else {
                alertDiv.classList.remove('show');
                setTimeout(() => alertDiv.remove(), 150); 
            }
        }, duration);
    }
}

function createStatsTable(statsData, title) {
    if (!statsData || typeof statsData !== 'object') {
        return `<p class="text-muted">${title}: Aucune donnée disponible.</p>`; // Traduit
    }
    let tableHtml = `<h6 class="mt-3"><i class="fas fa-chart-bar text-primary"></i> ${title}:</h6>`;
    tableHtml += '<table id="uploadInfoTable" class="table table-sm table-bordered table-striped small">';
    tableHtml += '<thead class="thead-light"><tr><th>Statistique</th><th>Valeur</th></tr></thead><tbody>'; // Traduit
    const statOrder = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'];
    statOrder.forEach(key => {
        if (statsData.hasOwnProperty(key)) {
            let value = statsData[key];
            if (typeof value === 'number') {
                value = (key === 'count') ? parseInt(value).toLocaleString('fr-FR') : parseFloat(value).toLocaleString('fr-FR', { minimumFractionDigits: 2, maximumFractionDigits: 4 });
            }
            tableHtml += `<tr><td>${key}</td><td>${value}</td></tr>`;
        }
    });
    tableHtml += '</tbody></table>';
    return tableHtml;
}

async function uploadFile() {
    const fileInput = document.getElementById('fileUpload');
    const uploadInfoDiv = document.getElementById('uploadInfo');
    const uploadInfoContainer = document.getElementById('uploadInfoContainer');
    const toggleBtn = document.getElementById('toggleUploadInfoBtn');
    
    if (fileInput.files.length === 0) {
        showAlert('Veuillez sélectionner un fichier.', 'warning'); // Traduit
        return;
    }
    showLoading('uploadInfo', 'Téléversement et traitement du fichier...'); // Traduit
    uploadInfoContainer.style.display = 'block'; 
    $('#uploadInfoDetails').collapse('show'); 
    if (toggleBtn) {
        toggleBtn.classList.remove('collapsed');
        toggleBtn.setAttribute('aria-expanded', 'true');
        const icon = toggleBtn.querySelector('i');
        if(icon) icon.className = 'fas fa-chevron-up';
    }


    isDataLoaded = false; 
    allMetricsData = null; 
    updateButtonStates();

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        const response = await fetch(`${API_BASE_URL}/upload`, {
            method: 'POST',
            body: formData,
        });
        const result = await response.json();
        uploadInfoDiv.innerHTML = ''; 

        if (response.ok) {
            showAlert('Fichier téléversé et traité avec succès !', 'success'); // Traduit
            isDataLoaded = true; 
            let generalInfoHtml = `<div class="row mb-2">
                                <div class="col-sm-6"><i class="fas fa-file-csv text-primary"></i> <strong>Fichier :</strong> ${fileInput.files[0].name}</div>
                                <div class="col-sm-6"><i class="fas fa-check-circle text-success"></i> <strong>Message :</strong> ${result.message}</div>
                                <div class="col-sm-6"><i class="fas fa-columns text-info"></i> <strong>Colonnes :</strong> ${result.data_info.columns.join(', ')}</div>
                                <div class="col-sm-6"><i class="far fa-calendar-alt text-info"></i> <strong>Plage de dates :</strong> ${result.data_info.start_date} à ${result.data_info.end_date}</div>
                                <div class="col-sm-6"><i class="fas fa-list-ol text-info"></i> <strong>Nombre de lignes :</strong> ${result.data_info.num_rows}</div>
                            </div>`;
            
            const statsTableHtml = createStatsTable(result.data_info.descriptive_stats, "Statistiques du Prix de Clôture"); // Traduit
            uploadInfoDiv.innerHTML = generalInfoHtml + statsTableHtml;
            
            localStorage.setItem('activeSectionTUNIndex', 'overview-section');
            document.querySelector('a[href="#overview-section"]').click();
            fetchDataOverview(); 
        } else {
            showAlert(`Erreur de téléversement : ${result.error}`, 'danger'); // Traduit
            uploadInfoDiv.innerHTML = `<p class="text-danger"><i class="fas fa-exclamation-triangle"></i> Erreur : ${result.error}</p>`;
            isDataLoaded = false;
        }
    } catch (error) {
        uploadInfoDiv.innerHTML = ''; 
        showAlert(`Erreur réseau lors du téléversement : ${error}`, 'danger'); // Traduit
        uploadInfoDiv.innerHTML = `<p class="text-danger"><i class="fas fa-ethernet"></i> Erreur Réseau : ${error}</p>`;
        isDataLoaded = false;
    } finally {
        updateButtonStates(); 
    }
}

async function fetchDataOverview() {
    const contentDiv = document.getElementById('dataOverviewContent'); 
    showLoading('dataOverviewContent', 'Chargement de l\'aperçu des données...'); // Traduit

    try {
        const response = await fetch(`${API_BASE_URL}/data/overview`);
        const result = await response.json();
        contentDiv.innerHTML = ''; 

        contentDiv.innerHTML = ` 
            <h4><i class="fas fa-calculator"></i> Statistiques Descriptives :</h4>
            <div id="descriptiveStatsTableContainerInner" class="table-responsive"></div>
            <hr>
            <h4><i class="fas fa-wave-square"></i> Évolution du Prix de Clôture :</h4>
            <div class="chart-container">
                 <img id="closePricePlotInner" src="#" alt="Graphique d'Évolution du Prix de Clôture" class="img-fluid" style="display:none;"/>
            </div>
        `;
        const newStatsContainer = document.getElementById('descriptiveStatsTableContainerInner');
        const newPlotImg = document.getElementById('closePricePlotInner');

        if (response.ok) {
            showAlert('Aperçu des données chargé avec succès.', 'success'); // Traduit
            let tableHtml = '<table class="table table-striped table-bordered table-sm table-hover">';
            tableHtml += `<thead class="thead-light"><tr>
                            <th data-toggle="tooltip" data-placement="top" title="Mesure statistique">Statistique</th>
                            <th data-toggle="tooltip" data-placement="top" title="Prix de clôture de l'indice">Clôture</th>
                            <th data-toggle="tooltip" data-placement="top" title="Prix d'ouverture de l'indice">Ouverture</th>
                            <th data-toggle="tooltip" data-placement="top" title="Prix le plus haut durant la période">Haut</th>
                            <th data-toggle="tooltip" data-placement="top" title="Prix le plus bas durant la période">Bas</th>
                            <th data-toggle="tooltip" data-placement="top" title="Volume des transactions (K pour milliers, M pour millions)">Volume</th>
                            <th data-toggle="tooltip" data-placement="top" title="Variation en pourcentage par rapport à la période précédente">Variation %</th>
                          </tr></thead>`;
            tableHtml += '<tbody>';

            const statKeys = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'];
            const statLabels = { 
                'count': {label: 'Nombre', tooltip: 'Nombre d\'observations'},
                'mean': {label: 'Moyenne', tooltip: 'Moyenne arithmétique'},
                'std': {label: 'Écart-type', tooltip: 'Écart-type, mesure de la dispersion des données'},
                'min': {label: 'Min', tooltip: 'Valeur minimale'},
                '25%': {label: 'Q1 (25%)', tooltip: 'Premier Quartile, 25% des données sont inférieures à cette valeur'},
                '50%': {label: 'Médiane (50%)', tooltip: 'Médiane, valeur centrale, 50% des données sont inférieures'},
                '75%': {label: 'Q3 (75%)', tooltip: 'Troisième Quartile, 75% des données sont inférieures à cette valeur'},
                'max': {label: 'Max', tooltip: 'Valeur maximale'}
            };

            const dataSources = ['close', 'open', 'high', 'low', 'volume', 'change'];
            
            for (const key of statKeys) {
                tableHtml += `<tr><td data-toggle="tooltip" data-placement="right" title="${statLabels[key].tooltip}"><strong>${statLabels[key].label}</strong></td>`;
                for (const dsKey of dataSources) {
                    const dataSource = result.stats[dsKey];
                    let value = '-';
                    if (dataSource && typeof dataSource === 'object' && dataSource[key] !== undefined && dataSource[key] !== null) {
                        value = (key === 'count') ? parseInt(dataSource[key]).toLocaleString('fr-FR') : parseFloat(dataSource[key]).toLocaleString('fr-FR', { minimumFractionDigits: 2, maximumFractionDigits: 4 });
                    }
                    tableHtml += `<td>${value}</td>`;
                }
                tableHtml += '</tr>';
            }
            tableHtml += '</tbody></table>';
            newStatsContainer.innerHTML = tableHtml;
            
            if (result.close_price_plot) {
                newPlotImg.src = `data:image/png;base64,${result.close_price_plot}`;
                newPlotImg.style.display = 'block';
            } else {
                newPlotImg.style.display = 'none';
            }
            isDataLoaded = true; 
            $('[data-toggle="tooltip"]').tooltip({ container: 'body' });
        } else {
            showAlert(`Erreur lors du chargement de l'aperçu : ${result.error}`, 'danger'); // Traduit
            newStatsContainer.innerHTML = `<p class="text-danger"><i class="fas fa-exclamation-circle"></i> Erreur de chargement des statistiques : ${result.error}</p>`;
            newPlotImg.style.display = 'none';
            isDataLoaded = false;
        }
    } catch (error) {
        contentDiv.innerHTML = `<p class="text-danger p-3"><i class="fas fa-ethernet"></i> Erreur réseau lors du chargement de l'aperçu : ${error}</p>`; // Traduit
        showAlert(`Erreur réseau : ${error}`, 'danger'); // Traduit
        isDataLoaded = false;
    } finally {
        updateButtonStates();
    }
}

function formatTestResultsToTable(testData, testName) {
    if (!testData || testData.error) {
        return `<p class="text-danger">${testName}: ${testData ? testData.error : 'Aucune donnée disponible ou erreur dans le test.'}</p>`; // Traduit
    }
    let resultClass = 'badge-secondary';
    if (testData.result === 'stationnaire') resultClass = 'badge-success';
    else if (testData.result === 'non stationnaire') resultClass = 'badge-warning';

    let adfPvalueTooltip = "P-value pour le Test ADF. Si < 0.05, la série est probablement stationnaire.";
    let kpssPvalueTooltip = "P-value pour le Test KPSS. Si > 0.05, la série est probablement stationnaire autour d'une tendance déterministe.";

    let table = `<h6 class="mt-2">${testName}: <span class="badge ${resultClass} p-1">${testData.result || 'N/A'}</span></h6>`;
    table += '<table class="table table-sm table-bordered table-striped small">';
    table += '<thead class="thead-light"><tr><th>Paramètre</th><th>Valeur</th></tr></thead>'; // Traduit
    table += '<tbody>'; 
    table += `<tr><td data-toggle="tooltip" data-placement="right" title="Valeur de la statistique du test.">Statistique</td><td>${parseFloat(testData.statistic).toFixed(4)}</td></tr>`; // Traduit
    table += `<tr><td data-toggle="tooltip" data-placement="right" title="${testName.includes('ADF') ? adfPvalueTooltip : kpssPvalueTooltip}">P-value</td><td>${parseFloat(testData.pvalue).toFixed(4)}</td></tr>`;
    if (testData.critical_values) {
        table += '<tr><td colspan="2"><strong>Valeurs Critiques :</strong></td></tr>'; // Traduit
        for (const cvKey in testData.critical_values) {
            table += `<tr><td>&nbsp;&nbsp;&nbsp;${cvKey}</td><td>${parseFloat(testData.critical_values[cvKey]).toFixed(4)}</td></tr>`;
        }
    }
    table += '</tbody></table>';
    return table;
}

async function fetchStationarityAnalysis() {
    const resultsDiv = document.getElementById('stationarityResults');
    const plotImg = document.getElementById('acfPacfPlot');
    showLoading('stationarityResults', 'Exécution des tests de stationnarité...'); // Traduit
    plotImg.style.display = 'none';
    plotImg.src = "#";

    try {
        const response = await fetch(`${API_BASE_URL}/analysis/stationarity`);
        const result = await response.json();
        resultsDiv.innerHTML = ''; 

        if (response.ok) {
            showAlert('Analyse de stationnarité terminée.', 'success'); // Traduit
            let html = formatTestResultsToTable(result.adf_test, 'Test ADF (Original)');
            html += formatTestResultsToTable(result.kpss_test, 'Test KPSS (Original)');
            html += formatTestResultsToTable(result.adf_diff_test, 'Test ADF (Différencié)');
            resultsDiv.innerHTML = html;
            if (result.acf_pacf_plot) {
                plotImg.src = `data:image/png;base64,${result.acf_pacf_plot}`;
                plotImg.style.display = 'block';
            }
            $('#stationarityResults [data-toggle="tooltip"]').tooltip({ container: 'body' });
        } else {
            showAlert(`Erreur dans l'analyse de stationnarité : ${result.error}`, 'danger'); // Traduit
            resultsDiv.innerHTML = `<p class="text-danger"><i class="fas fa-exclamation-circle"></i> Erreur : ${result.error}</p>`;
        }
    } catch (error) {
        resultsDiv.innerHTML = '';
        showAlert(`Erreur réseau : ${error}`, 'danger'); // Traduit
        resultsDiv.innerHTML = `<p class="text-danger"><i class="fas fa-ethernet"></i> Erreur Réseau : ${error}</p>`;
    }
}

async function fetchCorrelationAnalysis() {
    const matrixDiv = document.getElementById('correlationMatrix');
    const plotImg = document.getElementById('heatmapPlot');
    showLoading('correlationMatrix', 'Exécution de l\'analyse de corrélation...'); // Traduit
    plotImg.style.display = 'none';
    plotImg.src = "#";

    try {
        const response = await fetch(`${API_BASE_URL}/analysis/correlation`);
        const result = await response.json();
        matrixDiv.innerHTML = ''; 

        if (response.ok && result.correlation_matrix) {
            showAlert('Analyse de corrélation terminée.', 'success'); // Traduit
            let tableHtml = '<h6>Matrice de Corrélation :</h6>'; // Traduit
            tableHtml += '<table class="table table-sm table-bordered table-striped small table-hover">';
            
            const headers = Object.keys(result.correlation_matrix);
            tableHtml += '<thead class="thead-light"><tr><th data-toggle="tooltip" data-placement="top" title="Variables en corrélation">&nbsp;</th>'; // Traduit
            headers.forEach(header => tableHtml += `<th data-toggle="tooltip" data-placement="top" title="Corrélation avec ${header}">${header}</th>`);
            tableHtml += '</tr></thead><tbody>';

            headers.forEach(rowHeader => {
                tableHtml += `<tr><td><strong>${rowHeader}</strong></td>`;
                headers.forEach(colHeader => {
                    const value = result.correlation_matrix[rowHeader][colHeader];
                    let cellClass = '';
                    if (value > 0.7) cellClass = 'bg-success-light';
                    else if (value < -0.7) cellClass = 'bg-danger-light';
                    else if (value > 0.4) cellClass = 'bg-warning-light';
                    else if (value < -0.4) cellClass = 'bg-info-light';
                    tableHtml += `<td class="${cellClass}" data-toggle="tooltip" data-placement="top" title="Corrélation entre ${rowHeader} et ${colHeader}: ${parseFloat(value).toFixed(3)}">${parseFloat(value).toFixed(3)}</td>`;
                });
                tableHtml += '</tr>';
            });
            tableHtml += '</tbody></table>';
            matrixDiv.innerHTML = tableHtml;

            if (result.heatmap_plot) {
                plotImg.src = `data:image/png;base64,${result.heatmap_plot}`;
                plotImg.style.display = 'block';
            }
            $('#correlationMatrix [data-toggle="tooltip"]').tooltip({ container: 'body' });
        } else {
            showAlert(`Erreur dans l'analyse de corrélation : ${result.error || 'Matrice de corrélation non trouvée dans la réponse.'}`, 'danger'); // Traduit
            matrixDiv.innerHTML = `<p class="text-danger"><i class="fas fa-exclamation-circle"></i> Erreur : ${result.error || 'Matrice de corrélation non trouvée.'}</p>`;
        }
    } catch (error) {
        matrixDiv.innerHTML = '';
        showAlert(`Erreur réseau : ${error}`, 'danger'); // Traduit
        matrixDiv.innerHTML = `<p class="text-danger"><i class="fas fa-ethernet"></i> Erreur Réseau : ${error}</p>`;
    }
}

async function trainModels() {
    const statusDiv = document.getElementById('trainingStatus');
    const resultsDiv = document.getElementById('modelResults');
    showLoading('trainingStatus', 'Entraînement des modèles... Ceci peut prendre un moment.'); // Traduit
    resultsDiv.innerHTML = '';
    allMetricsData = null; 

    try {
        const response = await fetch(`${API_BASE_URL}/models/train`, { method: 'POST' });
        const result = await response.json();
        statusDiv.innerHTML = ''; 

        if (response.ok && result.status === 'success') {
            showAlert('Modèles entraînés avec succès ! Vérifiez les résultats et l\'évaluation.', 'success', 7000); // Traduit
            statusDiv.innerHTML = '<p class="text-success"><i class="fas fa-check-circle"></i> Modèles entraînés avec succès !</p>';
            let html = '<h5><i class="fas fa-clipboard-list"></i> Résumés d\'Entraînement & Paramètres :</h5>'; // Traduit
            for (const model in result.results) {
                html += `<details class="mb-2 border rounded p-2 shadow-sm"><summary class="font-weight-bold text-primary">${model}: <span class="badge badge-${result.results[model].status === 'success' ? 'success' : 'danger'}">${result.results[model].status}</span></summary>`;
                if (result.results[model].status === 'success') {
                    html += `<pre class="bg-light p-2 rounded small mt-1">${JSON.stringify(result.results[model], (key, value) => {
                        if (key === 'summary' && typeof value === 'string' && value.length > 300) {
                            return value.substring(0, 300) + "... (résumé tronqué)"; // Traduit
                        }
                        return value;
                    }, 2)}</pre>`;
                } else {
                    html += `<p class="text-danger mt-1"><i class="fas fa-exclamation-triangle"></i> Erreur : ${result.results[model].message}</p>`;
                }
                html += `</details>`;
            }
            resultsDiv.innerHTML = html;
            if (result.metrics) {
                allMetricsData = result.metrics;
            }
        } else {
            showAlert(`Erreur lors de l'entraînement des modèles : ${result.error || 'Erreur inconnue durant l\'entraînement.'}`, 'danger', 7000); // Traduit
            statusDiv.innerHTML = `<p class="text-danger"><i class="fas fa-times-circle"></i> Erreur d'entraînement des modèles : ${result.error || 'Erreur inconnue'}</p>`;
        }
    } catch (error) {
        statusDiv.innerHTML = '';
        showAlert(`Erreur réseau durant l'entraînement des modèles : ${error}`, 'danger'); // Traduit
        statusDiv.innerHTML = `<p class="text-danger"><i class="fas fa-ethernet"></i> Erreur Réseau : ${error}</p>`;
    }
}

function populateMetricFilters() {
    const filterModelSelect = document.getElementById('filterModel');
    const filterMetricNameSelect = document.getElementById('filterMetricName');

    filterModelSelect.length = 1; 
    filterMetricNameSelect.length = 1;

    if (!allMetricsData) return;

    const models = new Set();
    const metricNames = new Set();

    for (const setType of ['train', 'test']) {
        if (allMetricsData[setType]) {
            for (const modelName in allMetricsData[setType]) {
                models.add(modelName);
                for (const metricName in allMetricsData[setType][modelName]) {
                    metricNames.add(metricName);
                }
            }
        }
    }

    Array.from(models).sort().forEach(model => {
        const option = document.createElement('option');
        option.value = model;
        option.textContent = model;
        filterModelSelect.appendChild(option);
    });

    Array.from(metricNames).sort().forEach(metric => {
        const option = document.createElement('option');
        option.value = metric;
        option.textContent = metric;
        filterMetricNameSelect.appendChild(option);
    });
}


function renderMetricsTableWithFilters() {
    const metricsTableDiv = document.getElementById('evaluationMetricsTable');
    if (!allMetricsData) {
        metricsTableDiv.innerHTML = '<p class="text-muted text-center p-3">Aucune donnée de métrique disponible. Veuillez entraîner les modèles et cliquer sur "Évaluer Modèles & Métriques".</p>'; // Traduit
        return;
    }

    const filterModel = document.getElementById('filterModel').value;
    const filterSet = document.getElementById('filterSet').value;
    const filterMetricName = document.getElementById('filterMetricName').value;

    let tableHtml = '<table class="table table-sm table-bordered table-striped table-hover"><thead><tr><th>Modèle</th><th>Ensemble</th><th>Métrique</th><th>Valeur</th></tr></thead><tbody>'; // Traduit
    let hasRows = false;

    for (const setType of ['train', 'test']) {
        if (filterSet && filterSet !== setType) continue; 

        if (allMetricsData[setType]) {
            const sortedModelNames = Object.keys(allMetricsData[setType]).sort();
            for (const modelName of sortedModelNames) {
                if (filterModel && filterModel !== modelName) continue; 

                const sortedMetricNames = Object.keys(allMetricsData[setType][modelName]).sort();
                for (const metricName of sortedMetricNames) {
                    if (filterMetricName && filterMetricName !== metricName) continue; 

                    const value = allMetricsData[setType][modelName][metricName];
                    tableHtml += `<tr><td>${modelName}</td><td>${setType}</td><td>${metricName}</td><td>${value !== null && value !== undefined && !isNaN(value) ? parseFloat(value).toFixed(4) : 'N/A'}</td></tr>`;
                    hasRows = true;
                }
            }
        }
    }

    if (!hasRows) {
        tableHtml += '<tr><td colspan="4" class="text-center text-muted">Aucune métrique ne correspond aux filtres actuels.</td></tr>'; // Traduit
    }
    tableHtml += '</tbody></table>';
    metricsTableDiv.innerHTML = tableHtml;
}


async function evaluateModels() {
    const metricsTableDiv = document.getElementById('evaluationMetricsTable'); 
    const plotsDiv = document.getElementById('evaluationPlots');
    showLoading('evaluationMetricsTable', 'Chargement des métriques & graphiques d\'évaluation...'); // Traduit
    plotsDiv.innerHTML = ''; 

    try {
        const response = await fetch(`${API_BASE_URL}/models/evaluate`);
        const result = await response.json();
        
        if (response.ok) {
            showAlert('Données d\'évaluation des modèles chargées/mises à jour.', 'success'); // Traduit
            if (result.metrics) {
                allMetricsData = result.metrics; 
                populateMetricFilters();
                renderMetricsTableWithFilters(); 
            } else {
                 metricsTableDiv.innerHTML = '<p class="text-center text-muted p-3">Aucune donnée de métrique retournée par l\'évaluation.</p>'; // Traduit
            }

            let plotsHtml = '';
            if (result.plots && Object.keys(result.plots).length > 0) {
                for (const plotName in result.plots) {
                    if(result.plots[plotName]) {
                        const title = plotName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                        plotsHtml += `<div class="mt-3 border rounded p-2 shadow-sm"><h6><i class="fas fa-image"></i> ${title}</h6><img src="data:image/png;base64,${result.plots[plotName]}" class="img-fluid mb-2" alt="${title}"/></div>`;
                    }
                }
            } else {
                plotsHtml = '<p class="text-center text-muted p-3">Aucun graphique d\'évaluation disponible.</p>'; // Traduit
            }
            plotsDiv.innerHTML = plotsHtml; 
        } else {
            showAlert(`Erreur lors de l'évaluation des modèles : ${result.error}`, 'danger'); // Traduit
            metricsTableDiv.innerHTML = `<p class="text-danger"><i class="fas fa-exclamation-circle"></i> Erreur : ${result.error}</p>`;
            allMetricsData = null; 
            populateMetricFilters(); 
            renderMetricsTableWithFilters(); 
        }
    } catch (error) {
        metricsTableDiv.innerHTML = '';
        showAlert(`Erreur réseau durant l'évaluation des modèles : ${error}`, 'danger'); // Traduit
        metricsTableDiv.innerHTML = `<p class="text-danger"><i class="fas fa-ethernet"></i> Erreur Réseau : ${error}</p>`;
        allMetricsData = null;
        populateMetricFilters();
        renderMetricsTableWithFilters();
    }
}


async function makePrediction() {
    const modelSelect = document.getElementById('modelSelect').value;
    const stepsInput = document.getElementById('stepsInput').value;
    const predictionsDiv = document.getElementById('futurePredictions');
    showLoading('futurePredictions', `Génération de ${stepsInput} prédictions avec ${modelSelect}...`); // Traduit

    try {
        const response = await fetch(`${API_BASE_URL}/models/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: modelSelect, steps: parseInt(stepsInput) })
        });
        const result = await response.json();
        predictionsDiv.innerHTML = ''; 

        if (response.ok) {
            showAlert('Prédictions futures générées.', 'success'); // Traduit
            let html = `<h6><i class="fas fa-crystal-ball"></i> Prédictions de ${result.model}:</h6><ul class="list-group">`; // Traduit
            result.predictions.forEach((pred, index) => {
                html += `<li class="list-group-item d-flex justify-content-between align-items-center">
                            <span><i class="far fa-calendar-check"></i> Date: ${result.dates[index]}</span>
                            <span class="badge badge-primary badge-pill p-2">Clôture Prédite : ${parseFloat(pred).toFixed(2)}</span>
                         </li>`; // Traduit
            });
            html += '</ul>';
            predictionsDiv.innerHTML = html;
        } else {
            showAlert(`Erreur lors de la prédiction : ${result.error}`, 'danger'); // Traduit
            predictionsDiv.innerHTML = `<p class="text-danger"><i class="fas fa-exclamation-circle"></i> Erreur : ${result.error}</p>`;
        }
    } catch (error) {
        predictionsDiv.innerHTML = '';
        showAlert(`Erreur réseau durant la prédiction : ${error}`, 'danger'); // Traduit
        predictionsDiv.innerHTML = `<p class="text-danger"><i class="fas fa-ethernet"></i> Erreur Réseau : ${error}</p>`;
    }
}

async function exportData() {
    const format = document.getElementById('exportFormat').value;
    const includePredictions = document.getElementById('includePredictionsExport').checked;
    
    if (format === 'pdf') {
        showAlert("L'exportation PDF n'est pas encore entièrement implémentée. Un fichier CSV sera téléchargé à la place pour le moment.", 'warning', 7000); // Traduit
        // Pour l'instant, on force le CSV si PDF est sélectionné, car le backend ne gère pas encore le PDF.
        // Vous pouvez changer cela si vous implémentez la génération PDF côté backend.
        // format = 'csv'; // Décommenter pour forcer CSV
    }

    showAlert(`Préparation de l'exportation ${format.toUpperCase()}...`, 'info', null); 

    try {
        const response = await fetch(`${API_BASE_URL}/data/export`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ format: format, include_predictions: includePredictions })
        });

        const alertContainer = document.getElementById('alert-container');
        const stickyAlert = alertContainer.querySelector('.alert-info'); 
        if (stickyAlert && window.jQuery) $(stickyAlert).alert('close');
        else if (stickyAlert) stickyAlert.remove();


        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            const contentDisposition = response.headers.get('Content-Disposition');
            let filename = `tunindex_analyse.${format}`; 
            if (contentDisposition) {
                const filenameMatch = contentDisposition.match(/filename="?([^"]+)"?/);
                if (filenameMatch && filenameMatch.length > 1) {
                    filename = filenameMatch[1];
                }
            }
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            a.remove();
            showAlert(`Données exportées avec succès en tant que ${filename}.`, 'success'); // Traduit
        } else {
            const result = await response.json(); 
            showAlert(`Erreur lors de l'exportation des données : ${result.error}`, 'danger'); // Traduit
        }
    } catch (error) {
        const alertContainer = document.getElementById('alert-container');
        const stickyAlert = alertContainer.querySelector('.alert-info'); 
        if (stickyAlert && window.jQuery) $(stickyAlert).alert('close');
        else if (stickyAlert) stickyAlert.remove();
        showAlert(`Erreur réseau durant l'exportation : ${error}`, 'danger'); // Traduit
    }
}