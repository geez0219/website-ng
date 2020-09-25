import { Component, OnInit } from '@angular/core';
import { MatSnackBar } from '@angular/material/snack-bar';
import { Title, Meta } from '@angular/platform-browser';

import { SnackbarComponent } from '../snackbar/snackbar.component';
import { GlobalService } from '../global.service';
import { Router, ActivatedRoute } from '@angular/router'
import { HttpClient, HttpHeaders } from '@angular/common/http'

@Component({
  selector: 'app-install',
  templateUrl: './install.component.html',
  styleUrls: ['../api/api.component.css']
})
export class InstallComponent implements OnInit {
  installText: string;
  selectedVersion: string;

  VERSION = 'version';

  contentHeaderDict = {
    'Accept': 'application/json, text/plain',
    'Access-Control-Allow-Origin': '*'
  }

  contentRequestOptions = {
    responseType: 'text' as 'text',
    headers: new HttpHeaders(this.contentHeaderDict)
  };

  data = {
    name: 'Installing FastEstimator',
    bio: 'Installing FastEstimator',
  };

  durationInSeconds = 3;

  constructor(private _snackBar: MatSnackBar,
              private title: Title,
              private meta: Meta,
              private http: HttpClient,
              private router: Router,
              private route: ActivatedRoute,
              private globalService: GlobalService) { }

  ngOnInit() {
    this.route.params.subscribe(params => this.handleRouteChange(params));

    this.title.setTitle(this.data.name);
    this.meta.addTags([
      { name: 'og:url', content: '/install' + this.selectedVersion },
      { name: 'og:title', content: this.data.name },
      { name: 'og:description', content: this.data.bio },
    ]);
  }

  handleRouteChange(params) {
    this.selectedVersion = params[this.VERSION];
    this.getInstallText();
  }

  getInstallText() {
    this.http.get(
      'assets/branches/' + this.selectedVersion + '/install.md', this.contentRequestOptions).subscribe(data => {
      this.installText = data;
      this.globalService.resetLoading();
    },
      error => {
        console.error(error);
        this.globalService.resetLoading();
      });
  }

  copied(event: any) {
    this._snackBar.openFromComponent(SnackbarComponent, {
      duration: this.durationInSeconds * 1000,
      data: 'Copied'
    });
  }
}
