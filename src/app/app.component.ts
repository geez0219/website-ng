import { HttpClient } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';

import { Version } from './version';
import { GlobalService } from './global.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css'],
})
export class AppComponent implements OnInit {
  title = 'fastestimator-web';
  branchesLoaded: boolean;

  constructor(
    private http: HttpClient,
    private router: Router,
    private route: ActivatedRoute,
    private globalService: GlobalService
  ) {}

  ngOnInit() {
    this.http.get('assets/branches/branches.json').subscribe(
      (data) => {
        this.globalService.setVersions(data as Version[]);

        /* check if URL contains a versioned link already or not */
        const urlFragments: string[] = this.router.url.split('/');
        if (urlFragments.length >= 3) {
          const version = (data as Version[]).find(d => d.name === urlFragments[2]);
          this.globalService.setSelectedVersion(version);
          this.globalService.version.next(version.name);
        }

        this.branchesLoaded = true;
      },
      (error) => {
        console.error(error);
      }
    );
  }
}
