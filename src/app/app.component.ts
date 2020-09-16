import { HttpClient } from '@angular/common/http';
import { Component } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';

import { Branch } from './branch';
import { GlobalService } from './global.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'fastestimator-web';
  branchesLoaded: boolean;

  constructor(private http: HttpClient,
    private router: Router,
    private route: ActivatedRoute,
    private globalService: GlobalService) {
      this.http.get('assets//branches/branches.json').subscribe(data => {
        this.globalService.setBranches(<Branch[]>data);
        this.branchesLoaded = true;
      },
      error => {
        console.error(error);
      });
    }

}
